from abc import ABC, abstractmethod
from sklearn.datasets import fetch_openml
from torchvision.datasets import CIFAR10
from regimes import BaseRegime
import numpy as np


class AbstractDataGenerator(ABC):

    @abstractmethod
    def make_dataset(self, **kwargs):
        raise NotImplementedError("Abstract class shall not be implemented.")


class SyntheticDataGenerator(AbstractDataGenerator):

    def __init__(self, logger):
        super().__init__(logger)
        self.logger = logger

    def make_dataset(self, **kwargs):
        seed = kwargs.get("seed", 42)
        d = kwargs.get("d", 10)
        n = kwargs.get("n", 3000)
        n_test = kwargs.get("n_test", 600)
        m_star = kwargs.get("m_star", 20)
        gt_activation = kwargs.get("gt_activation", "relu")

        rng = np.random.default_rng(seed)

        self.logger.info("=" * 60)
        self.logger.info("DATA GENERATION")
        self.logger.info("=" * 60)
        self.logger.info(f"  d={d}, n={n}, n_test={n_test}, "
                         f"m*={m_star}, activation={gt_activation}")

        # Ground-truth network (fixed, not trained)
        gt = BaseRegime(
            d=d,
            m=m_star,
            activation=gt_activation,
            seed=seed
        )

        # X ~ N(0, I/d)
        cov = np.eye(d) / d
        X_train = rng.multivariate_normal(np.zeros(d), cov, size=n)
        X_test = rng.multivariate_normal(np.zeros(d), cov, size=n_test)

        y_train = gt.forward(X_train)
        y_test = gt.forward(X_test)

        self.logger.info(f"  X_train: {X_train.shape}   y_train: {y_train.shape}")
        self.logger.info(f"  X_test : {X_test.shape}    y_test : {y_test.shape}")
        self.logger.info(f"  y_train — mean={y_train.mean():.4f}, std={y_train.std():.4f}")
        self.logger.info(f"  Null MSE (predict 0): {float(np.mean(y_test ** 2)):.6f}")

        return X_train, y_train, X_test, y_test


class MNISTDataGenerator(AbstractDataGenerator):

    def __init__(self, logger):
        super().__init__(logger)
        self.logger = logger

    def make_dataset(self, **kwargs):
        seed = kwargs.get("seed", 42)
        classes = kwargs.get("classes", (0, 1))  # Binary classification
        n_train = kwargs.get("n", 5000)
        n_test = kwargs.get("n_test", 1000)
        normalize = kwargs.get("normalize", True)

        rng = np.random.default_rng(seed)

        self.logger.info("=" * 60)
        self.logger.info("MNIST DATA GENERATION")
        self.logger.info("=" * 60)
        self.logger.info(f"  classes={classes}, n_train={n_train}, n_test={n_test}")

        X, y = fetch_openml(
            "mnist_784",
            version=1,
            return_X_y=True,
            as_frame=False
        )
        y = y.astype(int)

        # Binary classification
        mask = np.isin(y, classes)
        X = X[mask]
        y = y[mask]
        y = (y == classes[1]).astype(np.float64)

        # Shuffle
        permutation = rng.permutation(len(X))
        X = X[permutation]
        y = y[permutation]

        # Train test split
        if n_train + n_test > len(X):
            raise ValueError(
                f"Requested n_train + n_test = {n_train + n_test}, "
                f"but only {len(X)} samples available for classes {classes}."
            )

        X_train = X[:n_train]
        y_train = y[:n_train]
        X_test = X[n_train:n_train + n_test]
        y_test = y[n_train:n_train + n_test]

        # Scale
        X_train = X_train.astype(np.float64) / 255.0
        X_test = X_test.astype(np.float64) / 255.0

        # Normalization
        if normalize:
            # Calculate mean and std from training data
            mean = X_train.mean(axis=0, keepdims=True)
            std = X_train.std(axis=0, keepdims=True)
            std[std < 1e-8] = 1.0
            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std

        self.logger.info(f"  X_train: {X_train.shape}   y_train: {y_train.shape}")
        self.logger.info(f"  X_test : {X_test.shape}    y_test : {y_test.shape}")
        self.logger.info(f"  Train label mean: {y_train.mean():.4f}")
        self.logger.info(f"  Test label mean : {y_test.mean():.4f}")

        return X_train, y_train, X_test, y_test


class CIFAR10DataGenerator(AbstractDataGenerator):

    def __init__(self, logger):
        super().__init__(logger)
        self.logger = logger

    def make_dataset(self, **kwargs):
        seed = kwargs.get("seed", 42)
        classes = kwargs.get("classes", (0, 1))  # Binary classification
        n_train = kwargs.get("n", 5000)
        n_test = kwargs.get("n_test", 1000)
        normalize = kwargs.get("normalize", True)
        root = kwargs.get("root", "./data")

        rng = np.random.default_rng(seed)

        self.logger.info("=" * 60)
        self.logger.info("CIFAR-10 DATA GENERATION")
        self.logger.info("=" * 60)
        self.logger.info(f"  classes={classes}, n_train={n_train}, n_test={n_test}")

        train_dataset = CIFAR10(
            root=root,
            train=True,
            download=True
        )
        test_dataset = CIFAR10(
            root=root,
            train=False,
            download=True
        )

        X_train = train_dataset.data
        y_train = np.array(train_dataset.targets)

        X_test = test_dataset.data
        y_test = np.array(test_dataset.targets)

        # Binary classification
        train_mask = np.isin(y_train, classes)
        test_mask = np.isin(y_test, classes)

        X_train = X_train[train_mask]
        y_train = y_train[train_mask]

        X_test = X_test[test_mask]
        y_test = y_test[test_mask]

        y_train = (y_train == classes[1]).astype(np.float64)
        y_test = (y_test == classes[1]).astype(np.float64)

        # Shuffle
        permutation_train = rng.permutation(len(X_train))
        permutation_test = rng.permutation(len(X_test))
        X_train = X_train[permutation_train]
        y_train = y_train[permutation_train]
        X_test = X_test[permutation_test]
        y_test = y_test[permutation_test]

        # Train test split
        if n_train > len(X_train):
            raise ValueError(
                f"Requested n_train={n_train}, but only {len(X_train)} "
                f"training samples available for classes {classes}."
            )
        if n_test > len(X_test):
            raise ValueError(
                f"Requested n_test={n_test}, but only {len(X_test)} "
                f"test samples available for classes {classes}."
            )

        X_train = X_train[:n_train]
        y_train = y_train[:n_train]
        X_test = X_test[:n_test]
        y_test = y_test[:n_test]

        # Flatten and scale
        X_train = X_train.astype(np.float64).reshape(len(X_train), -1) / 255.0
        X_test = X_test.astype(np.float64).reshape(len(X_test), -1) / 255.0

        # Normalization
        if normalize:
            # Calculate mean and std from training data
            mean = X_train.mean(axis=0, keepdims=True)
            std = X_train.std(axis=0, keepdims=True)
            std[std < 1e-8] = 1.0
            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std

        self.logger.info(f"  X_train: {X_train.shape}   y_train: {y_train.shape}")
        self.logger.info(f"  X_test : {X_test.shape}    y_test : {y_test.shape}")
        self.logger.info(f"  Train label mean: {y_train.mean():.4f}")
        self.logger.info(f"  Test label mean : {y_test.mean():.4f}")

        return X_train, y_train, X_test, y_test
