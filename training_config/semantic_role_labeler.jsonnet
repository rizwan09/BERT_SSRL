// Configuration for a semantic role labeler model based on:
//   He, Luheng et al. “Deep Semantic Role Labeling: What Works and What's Next.” ACL (2017).
{
  "dataset_reader":{"type":"srl"},
  "train_data_path": "/home/rizwan/SBCR/data/conll12/conll-formatted-ontonotes-5.0/data/train/data/english/",
  "validation_data_path": "/home/rizwan/SBCR/data/conll12/conll-formatted-ontonotes-5.0/data/development/data/english/",
  "model": {
    "type": "srl_bert",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 100,
            "pretrained_file": "/home/rizwan/.allennlp/cache/glove.6B.100d.txt.gz",
            "trainable": true
        }
      }
    },
    "initializer": [
      [
        "tag_projection_layer.*weight",
        {
          "type": "orthogonal"
        }
      ]
    ],
    "encoder": {
      "type": "alternating_lstm",
      "input_size": 200,
      "hidden_size": 300,
      "num_layers": 8,
      "recurrent_dropout_probability": 0.1,
      "use_highway": true
    },
    "binary_feature_dim": 100
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 80
  },

  "trainer": {
    "num_epochs": 500,
    "grad_clipping": 1.0,
    "patience": 20,
    "validation_metric": "+f1-measure-overall",
    "cuda_device": [0,1,2,3],
    "optimizer": {
      "type": "adadelta",
      "rho": 0.95
    }
  }
}
