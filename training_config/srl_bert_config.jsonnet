{
    "dataset_reader": {
        "type": "srl",
        "bert_model_name": "bert-base-uncased"
    },
    "iterator": {
        "type": "bucket",
        "batch_size": 32,
        "sorting_keys": [
            [
                "tokens",
                "num_tokens"
            ]
        ]
    },
    "model": {
        "type": "srl_bert",
        "bert_model": "bert-base-uncased",
        "embedding_dropout": 0.1
    },
    "train_data_path": "/home/rizwan/SBCR/data/conll12/conll-formatted-ontonotes-5.0/data/train/",
    "validation_data_path": "//home/rizwan/SBCR/data/conll12/conll-formatted-ontonotes-5.0/data/development/",
    "trainer": {
        "cuda_device": [0,1,2,3],
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 15,
            "num_steps_per_epoch": 8829
        },
        "num_epochs": 15,
        "num_serialized_models_to_keep": 2,
        "optimizer": {
            "type": "bert_adam",
            "lr": 2e-05,
            "max_grad_norm": 1,
            "parameter_groups": [
                [
                    [
                        "bias",
                        "LayerNorm.bias",
                        "LayerNorm.weight",
                        "layer_norm.weight"
                    ],
                    {
                        "weight_decay": 0
                    }
                ]
            ],
            "t_total": -1,
            "weight_decay": 0.01
        },
        "should_log_learning_rate": true,
        "validation_metric": "+f1-measure-overall"
    }
}