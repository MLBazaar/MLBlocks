import random


class Type(object):
    """Hyperparameter types."""
    INT = "int"
    INT_EXP = "int_exp"
    INT_CAT = "int_cat"
    FLOAT = "float"
    FLOAT_EXP = "float_exp"
    FLOAT_CAT = "float_cat"
    STRING = "string"
    BOOL = "bool"


_CATEGORICAL_TYPES = (Type.STRING, Type.BOOL, Type.INT_CAT, Type.FLOAT_CAT)
_INTEGER_TYPES = (Type.INT, Type.INT_EXP)
_FLOAT_TYPES = (Type.FLOAT, Type.FLOAT_EXP)


class MLHyperparam(object):
    """A Hyperparameter that the DeepMining system can act on.

    Should belong to a MLBLock.

    Attributes:
        param_name: The name of this hyperparameter.
        block_name: The name of the MLBLock this hyperparameter belongs
            to.
        param_type: The type of this hyperparameter. See the Type object
            in this module for possible types.
        param_range: A list of the form [a, b] such that the value of
            this hyperparameter is between a and b inclusive. If this
            hyperparameter is categorical, this is a list containing all
            possible values of this hyperparameter.
        is_conditional: If this hyperparameter is conditional. Defaults
            to false (conditional).
        value: The current value of this hyperparameter.
    """

    def __init__(self, param_name, param_type, param_range,
                 is_conditional=False, value=None):
        """Initialize this Hyperparameter.

        Args:
            param_name: The name of this Hyperparameter.
            param_type: The type of this Hyperparameter.
            param_range: A list of the form [a, b] such that the value
                of this hyperparameter is between a and b inclusive. If
                this hyperparameter is categorical, this is a list
                containing all possible values of this hyperparameter.
            is_conditional: If this hyperparameter is conditional.
                Defaults to false (not conditional).
            value: The value to initialize this hyperparameter to.
                Randomly initialized in the specified param_range if not
                specified.
        """
        self.param_name = param_name
        self.block_name = None

        self.param_type = param_type
        self.param_range = param_range

        # This attribute exists for now to mark conditional
        # hyperparameters to not be tuned.
        self.is_conditional = is_conditional

        # Value randomly initialized in range if not specified.
        self._value = value if value is not None else self._random_init()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if self.param_type in _INTEGER_TYPES:
            self._value = int(value)

        elif self.param_type in _FLOAT_TYPES:
            self._value = float(value)

        else:
            self._value = value

    def _random_init(self):
        """Initialize using random values within param_range."""
        # Strings and bools should always be categorical
        if self.param_type in _CATEGORICAL_TYPES:
            value = random.choice(self.param_range)

        elif self.param_type in _INTEGER_TYPES:
            value = random.randint(*self.param_range)

        elif self.param_type in _FLOAT_TYPES:
            value = random.uniform(*self.param_range)

        else:
            raise AttributeError("Unexpected parameter type:", self.param_type)

        return value

    def __str__(self):
        return (
            "Hyperparameter: "
            "Name: {0}, "
            "Block Name: {1}, "
            "Type: {2}, "
            "Range: {3}, "
            "Value: {4}"
        ).format(
            self.param_name,
            self.block_name,
            self.param_type,
            self.param_range,
            self.value
        )
