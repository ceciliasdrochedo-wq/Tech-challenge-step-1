def test_src_package_is_importable():
    import src

    assert src is not None


def test_all_modules_are_importable():
    import src.config
    import src.data.loader
    import src.models.mlp
    import src.models.mlp_trainer
    import src.models.registry
    import src.models.trainer
    import src.pipeline
    import src.service.mlflow_service

    assert src.config is not None
    assert src.data.loader is not None
    assert src.models.mlp is not None
    assert src.models.mlp_trainer is not None
    assert src.models.trainer is not None
    assert src.models.registry is not None
    assert src.pipeline is not None
    assert src.service.mlflow_service is not None
