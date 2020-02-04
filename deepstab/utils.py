import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.nn.functional import conv2d
from tqdm import tqdm


# generates superresolution mask
# mask = np.zeros((32, 32))
# for x in range(32):
#     for y in range(32):
#         if (x + y) % 2 == 0:
#             mask[x, y] = 255


def dilate_torch(tensor, size):
    structuring_element = torch.ones((size, size)).view(1, 1, size, size).cuda()
    return (conv2d(tensor, structuring_element, stride=1, padding=(size // 2, size // 2)) > 0).float()


def cutout_mask(frame, mask, dilate_mask=True):
    if dilate_mask:
        mask = mask.filter(ImageFilter.MaxFilter(3))
    frame = np.array(frame)
    mask = np.array(mask)
    mask = 255 - mask
    frame[mask == 255] = 255
    return Image.fromarray(frame)


def bbox(mask):
    ImageDraw.Draw(mask).rectangle(mask.getbbox(), fill='white')
    return mask


def dilate(mask, size=3):
    return mask.filter(ImageFilter.MaxFilter(size))


def mask_tensor(x, m):
    return x * m + (1 - m)


def normalize(x):
    return x * 2 - 1


def denormalize(x):
    return (x + 1) / 2


def extract_mask(image, i=1, mode='L'):
    image = np.array(image)
    mask = np.ones(image.shape, dtype=np.uint8) * 255
    mask[image == i] = 0
    return Image.fromarray(mask).convert(mode)


def save_frames(save_dir, frames):
    for i, frame in enumerate(frames):
        save_path = os.path.join(save_dir, '{:05d}'.format(i))
        if isinstance(frame, Image.Image):
            frame.save(save_path)
        else:
            ValueError('Unhandled frame class')


def tensor_to_cv_image(image_tensor: torch.Tensor):
    return image_tensor.permute(1, 2, 0).numpy().astype(np.uint8)


def cv_image_to_tensor(mat: np.ndarray):
    return torch.from_numpy(mat).permute(2, 0, 1).float()


def pil_to_cv(pil_image):
    return np.array(pil_image)[:, :, ::-1]


def cv_to_pil(mat):
    return Image.fromarray(mat[:, :, ::-1])


class Progbar(object):
    """Displays a progress bar.
    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.
        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


class TQDMProgressBar(object):
    def __init__(self, epochs, steps):
        self.epoch = 1
        self.epochs = epochs
        self.steps = steps
        self.progress_bar = tqdm(total=steps, unit='step')
        self.observed = {}

    def add(self, name, value):
        if name in self.observed:
            self.observed[name].append(value)
        else:
            self.observed[name] = [value]

    def next_epoch(self):
        self.progress_bar.close()
        if self.epoch < self.epochs:
            self.progress_bar = tqdm(total=self.steps, unit='step')
        self.epoch += 1
        self.observed = {}

    def next_step(self):
        self.progress_bar.update()

    def update(self):
        description = f'Epoch: {self.epoch} - '
        for k, v in self.observed.items():
            description += f'{k}: {np.mean(v):.3f} '
        self.progress_bar.set_description(description)


def logs_to_writer(writer, logs, epoch):
    name_to_values = {}
    for name, value in logs:
        if name in name_to_values:
            name_to_values[name].append(value)
        else:
            name_to_values[name] = [value]

    for name, values in name_to_values.items():
        if 'val_' in name:
            name = name.replace('val_', '') + '/val'
        else:
            name = name + '/train'
        writer.add_scalar(name, np.mean(values), epoch)


def cycle(iterable):
    while True:
        for x in iter(iterable):
            yield x


if __name__ == '__main__':
    img = Image.open('../data/raw/video/DAVIS/JPEGImages/480p/breakdance/00000.jpg')
    from torchvision.transforms import transforms

    mask = transforms.Resize((32, 32))(
        extract_mask(Image.open('../data/raw/video/DAVIS/Annotations_unsupervised/480p/camel/00000.png')))
    # res = cutout_mask(img, mask)
    # img.save('image.jpeg')
    import PIL.ImageOps

    PIL.ImageOps.invert(mask).save('mask.jpeg')
    # res.save('masked_image.jpeg')
