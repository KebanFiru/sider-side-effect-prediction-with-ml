import deepchem as dc


def load_sider(root: str = "data/raw", split: str = "scaffold"):

    tasks, datasets, _transformers = dc.molnet.load_sider(
        featurizer="Raw", splitter=split, data_dir=root, save_dir=root
    )
    train, valid, test = datasets
    return tasks, train, valid, test