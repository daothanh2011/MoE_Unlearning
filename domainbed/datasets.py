# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os

import numpy as np
import torch
import torchvision.datasets.folder
from PIL import Image, ImageFile
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    "CIFAR10",
    "CIFAR100",
    # Big images
    "CUB",
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW"
]


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001  # Default, subclasses may override
    CHECKPOINT_FREQ = 100  # Default, subclasses may override
    N_WORKERS = 8  # Default, subclasses may override
    ENVIRONMENTS = None  # Subclasses should override
    INPUT_SHAPE = None  # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )


class Debug28(Debug):
    N_WORKERS = 0
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']


class Debug224(Debug):
    N_WORKERS = 0
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                           self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class ColoredMNIST_E(MultipleEnvironmentMNIST):
    """ColoredMNIST with a configurable number of training environments E.

    Protocol from Wang et al., "Lost Domain Generalization Is a Natural
    Consequence of Lack of Training Domains", AAAI 2024.
    """
    # Class-level placeholder of length default_E + 1 = 9, so that
    # datasets.num_environments("ColoredMNIST_E") works before the dataset
    # is instantiated (e.g. in test_datasets.py). The real, p_e-annotated
    # names are written to self.ENVIRONMENTS in __init__.
    ENVIRONMENTS = [f'env_{i}' for i in range(9)]

    def __init__(self, root, test_envs, hparams):
        E = int(hparams.get('num_environments', 8))
        if E < 2:
            raise ValueError(f"num_environments must be >= 2, got {E}")

        candidates = np.linspace(1.0 / (E + 2), (E + 1) / (E + 2), E + 1)
        drop_idx = int(np.argmin(np.abs(candidates - 0.5)))
        train_p = np.delete(candidates, drop_idx).tolist()

        environments = train_p + [0.5]
        self.ENVIRONMENTS = [f'p={p:.3f}' for p in environments]

        super().__init__(root, environments, self.color_dataset,
                         (2, 28, 28,), 2)
        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        labels = (labels < 5).float()
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))),
               (1 - colors).long(), :, :] *= 0
        x = images.float().div_(255.0)
        y = labels.view(-1).long()
        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class ColoredMNIST_K(MultipleDomainDataset):
    """ColoredMNIST with K source domains + 1 fixed test domain (p=0.5).

    Sample budget (fixed across K):
      - test env: exactly `test_size` samples (default 10000), p_color = 0.5
      - source envs: remaining (70000 - test_size) split equally across K
        (any remainder < K is dropped)

    Source p_color values: linspace(1/(K+2), (K+1)/(K+2), K+1) with the value
    closest to 0.5 dropped. Test env appended at index K (last).
    """
    ENVIRONMENTS = [f'env_{i}' for i in range(10)]

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')
        K = int(hparams.get('num_source_domains', 9))
        if K < 2:
            raise ValueError(f"num_source_domains must be >= 2, got {K}")
        test_size = int(hparams.get('test_size', 10000))

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)
        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))
        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        N_total = len(original_images)
        if test_size >= N_total:
            raise ValueError(
                f"test_size ({test_size}) must be < total MNIST ({N_total})")
        N_train = N_total - test_size
        per_env = N_train // K
        if per_env < 1:
            raise ValueError(
                f"K={K} too large for source pool of {N_train} samples")

        shuffle = torch.randperm(N_total)
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        train_imgs = original_images[: K * per_env]
        train_lbls = original_labels[: K * per_env]
        test_imgs = original_images[N_train: N_train + test_size]
        test_lbls = original_labels[N_train: N_train + test_size]

        candidates = np.linspace(1.0 / (K + 2), (K + 1) / (K + 2), K + 1)
        drop_idx = int(np.argmin(np.abs(candidates - 0.5)))
        train_p = np.delete(candidates, drop_idx).tolist()
        self.ENVIRONMENTS = [f'p={p:.3f}' for p in train_p] + ['p=0.500']

        self.datasets = []
        for i, p in enumerate(train_p):
            ei = train_imgs[i * per_env: (i + 1) * per_env]
            el = train_lbls[i * per_env: (i + 1) * per_env]
            self.datasets.append(self.color_dataset(ei, el, p))
        self.datasets.append(self.color_dataset(test_imgs, test_lbls, 0.5))

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2


    def color_dataset(self, images, labels, environment):
        labels = (labels < 5).float()
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))),
               (1 - colors).long(), :, :] *= 0
        x = images.float().div_(255.0)
        y = labels.view(-1).long()
        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                                               interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class RotatedColoredMNIST_K(MultipleDomainDataset):
    """Rotated + Colored MNIST: K envs laid out row-major in a 13x10 grid.

    User specifies `num_source_domains` (= K, total env count). Envs are
    placed in a (rotation, color) grid in **row-major order**: each row
    fills all 10 colors before moving to the next rotation. The final row
    may be partial.

    Layout for K = 10*r + s   (where r = K//10, s = K - 10*r):
        - r FULL rows: rotations [0, 15, ..., 15*(r-1)],
          each with all 10 colors [0.0, 0.1, ..., 0.9]
        - 1 PARTIAL row (only if s > 0): rotation 15*r,
          with the first s colors [0.0, 0.1, ..., 0.1*(s-1)]

    Examples:
        K=3   -> 1 partial row:  rot=0, colors [0.0, 0.1, 0.2]
        K=10  -> 1 full row:     rot=0, colors [0.0..0.9]
        K=11  -> 1 full + 1 partial: rot=0 (10 colors) + rot=15 (color 0.0)
        K=25  -> 2 full + 1 partial: rot=0,15 (10 colors each) + rot=30 (5 colors)
        K=130 -> 13 full rows (max)

    Env at index `idx` has:
        rot   = 15 * (idx // 10)
        p     = 0.1 * (idx % 10)

    Sample budget: full MNIST (70000 samples) shuffled and split equally
    across the K envs. The user selects test env(s) externally via
    DomainBed's `test_envs` argument.

    Constraint: 1 <= K <= 130.
    """
    ENVIRONMENTS = [f'env_{k}' for k in range(130)]  # placeholder

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        K = int(hparams.get('num_source_domains', 9))
        if K < 1 or K > 130:
            raise ValueError(f"num_source_domains must be in [1, 130], got {K}")

        # ---- Load full MNIST (train + test pooled) ----
        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)
        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))
        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        N_total = len(original_images)
        per_env = N_total // K
        if per_env < 1:
            raise ValueError(
                f"K={K} too large for MNIST pool of {N_total} samples")

        shuffle = torch.randperm(N_total)
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]
        original_images = original_images[: K * per_env]
        original_labels = original_labels[: K * per_env]

        # ---- Build envs in row-major order ----
        # idx -> (rot_idx = idx // 10, color_idx = idx % 10)
        env_names = []
        self.datasets = []
        for idx in range(K):
            ri = idx // 10
            ci = idx % 10
            angle = 15.0 * ri
            p = 0.1 * ci
            ei = original_images[idx * per_env: (idx + 1) * per_env]
            el = original_labels[idx * per_env: (idx + 1) * per_env]
            self.datasets.append(self._make_env(ei, el, angle, p))
            env_names.append(f'rot={angle:.1f},p={p:.3f}')
        self.ENVIRONMENTS = env_names

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    # ------------------------------------------------------------------
    # Env construction: rotate first (on grayscale), then color
    # ------------------------------------------------------------------
    def _make_env(self, images, labels, angle, p_color):
        rotated = self._rotate_images(images, angle)  # float [N, 28, 28], in [0,255]

        labels = (labels < 5).float()
        labels = self._torch_xor(labels, self._torch_bernoulli(0.25, len(labels)))
        colors = self._torch_xor(labels, self._torch_bernoulli(p_color, len(labels)))

        imgs2 = torch.stack([rotated, rotated], dim=1)  # [N, 2, 28, 28]
        imgs2[torch.arange(len(imgs2)), (1 - colors).long(), :, :] *= 0

        x = imgs2.float().div_(255.0)
        y = labels.view(-1).long()
        return TensorDataset(x, y)

    def _rotate_images(self, images, angle):
        """Rotate [N, 28, 28] uint8 tensor by `angle` degrees. Returns float
        tensor [N, 28, 28] in [0, 255] range; downstream div_(255.0) in
        _make_env produces values in [0, 1]."""
        if angle == 0.0:
            return images.float()
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(
                x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor(),
        ])
        out = torch.zeros(len(images), 28, 28)
        for k in range(len(images)):
            out[k] = rotation(images[k]).squeeze(0) * 255.0
        return out

    def _torch_bernoulli(self, p, size):
        return (torch.rand(size) < p).float()

    def _torch_xor(self, a, b):
        return (a - b).abs()


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path, transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)


class CUB(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["Candy", "Mosaic", "Natural", "Udnie"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "CUB_DG/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class _PACSArrowEnv(torch.utils.data.Dataset):
    """A single PACS domain backed by a HuggingFace Arrow file.

    Exposes ``classes`` so callers that compute ``num_classes`` via
    ``len(env.classes)`` (as ``MultipleEnvironmentImageFolder`` does) keep
    working unchanged.
    """

    def __init__(self, hf_dataset, indices, classes, transform):
        self.hf_dataset = hf_dataset
        self.indices = indices
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.hf_dataset[int(self.indices[idx])]
        img = row["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"])


class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    # Maps the canonical ENVIRONMENTS letters to the `domain` string used
    # by the flwrlabs/pacs HuggingFace dataset. Index order here must match
    # ENVIRONMENTS so that ``test_envs`` indices stay consistent across
    # ImageFolder- and Arrow-backed instantiations.
    _HF_DOMAIN_ORDER = ["art_painting", "cartoon", "photo", "sketch"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        arrow_path = os.path.join(self.dir, "pacs-train.arrow")
        if os.path.isfile(arrow_path):
            MultipleDomainDataset.__init__(self)
            self._init_from_arrow(
                arrow_path, test_envs, hparams.get("data_augmentation", True))
        else:
            super().__init__(self.dir, test_envs,
                             hparams['data_augmentation'], hparams)

    @staticmethod
    def _import_hf_dataset():
        """Import HuggingFace ``datasets.Dataset`` despite sys.path shadowing.

        Other modules in this repo (notably ``domainbed/algorithms.py``)
        prepend the ``domainbed/`` directory to ``sys.path`` so that
        ``import vision_transformer`` works as a top-level name. That makes
        a plain ``from datasets import Dataset`` resolve to this very file
        (``domainbed/datasets.py``) instead of the installed HuggingFace
        package. We temporarily strip any sys.path entry pointing at this
        file's directory, and clear a possibly-poisoned ``sys.modules``
        entry, then perform the import.
        """
        import sys
        this_dir = os.path.dirname(os.path.abspath(__file__))
        saved_path = sys.path[:]
        saved_mod = sys.modules.pop('datasets', None)
        sys.path = [p for p in sys.path
                    if not p or os.path.abspath(p) != this_dir]
        try:
            from datasets import Dataset as HFDataset  # type: ignore
            return HFDataset
        except ImportError as e:
            raise ImportError(
                "Loading PACS from an Arrow file requires the `datasets` "
                "package. Install it with `pip install datasets`."
            ) from e
        finally:
            sys.path = saved_path
            # Only restore the previous binding if it was a real, distinct
            # module (i.e. not a stale reference to this file).
            if saved_mod is not None and getattr(
                    saved_mod, '__file__', None) != os.path.abspath(__file__):
                sys.modules['datasets'] = saved_mod

    def _init_from_arrow(self, arrow_path, test_envs, augment):
        HFDataset = self._import_hf_dataset()

        hf = HFDataset.from_file(arrow_path)
        required = {"image", "domain", "label"}
        if not required.issubset(hf.features.keys()):
            raise ValueError(
                f"Unexpected schema in {arrow_path}; expected "
                f"image/domain/label, got {list(hf.features.keys())}")
        label_names = list(hf.features["label"].names)

        env_indices = {d: [] for d in self._HF_DOMAIN_ORDER}
        for i, d in enumerate(hf["domain"]):
            if d in env_indices:
                env_indices[d].append(i)

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
        augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            normalize,
        ])

        self.datasets = []
        for i, domain in enumerate(self._HF_DOMAIN_ORDER):
            env_t = augment_transform if (augment and i not in test_envs) else base_transform
            self.datasets.append(
                _PACSArrowEnv(hf, env_indices[domain], label_names, env_t))

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(label_names)


class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 500
    N_STEPS = 15001
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class TerraIncognita(MultipleEnvironmentImageFolder):
    # may need larger weight decay
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)

    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = ["hospital_0", "hospital_1", "hospital_2", "hospital_3",
                    "hospital_4"]

    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = ["region_0", "region_1", "region_2", "region_3",
                    "region_4", "region_5"]

    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)


class _CIFARBase(MultipleDomainDataset):
    """Single-domain CIFAR for LoTUS-compatible unlearning (train/test splits in splits.py)."""

    INPUT_SHAPE = (3, 32, 32)
    N_STEPS = 20001
    CHECKPOINT_FREQ = 500
    N_WORKERS = 4
    ENVIRONMENTS = ["default"]

    def __init__(self, root, test_envs, hparams, tv_class, num_classes):
        super().__init__()
        if root is None:
            raise ValueError("Data directory not specified!")
        self.train_dataset = tv_class(
            root=root, train=True, download=True, transform=None,
        )
        self.test_dataset = tv_class(
            root=root, train=False, download=True, transform=None,
        )
        self.datasets = [self.train_dataset]
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = num_classes


class CIFAR10(_CIFARBase):
    def __init__(self, root, test_envs, hparams):
        from torchvision.datasets import CIFAR10 as CIFAR10TV
        super().__init__(root, test_envs, hparams, CIFAR10TV, 10)


class CIFAR100(_CIFARBase):
    def __init__(self, root, test_envs, hparams):
        from torchvision.datasets import CIFAR100 as CIFAR100TV
        super().__init__(root, test_envs, hparams, CIFAR100TV, 100)
