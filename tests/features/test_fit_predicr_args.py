from mlblocks.mlpipeline import MLPipeline


def test_fit_predict_args_in_init():

    def add(a, b):
        return a + b

    primitive = {
        'name': 'add',
        'primitive': add,
        'produce': {
            'args': [
                {
                    'name': 'a',
                    'type': 'float',
                },
                {
                    'name': 'b',
                    'type': 'float',
                },
            ],
            'output': [
                {
                    'type': 'float',
                    'name': 'out'
                }
            ]
        }
    }

    primitives = [primitive]
    init_params = {
        'add': {
            'b': 10
        }
    }
    pipeline = MLPipeline(primitives, init_params=init_params)

    out = pipeline.predict(a=3)

    assert out == 13
