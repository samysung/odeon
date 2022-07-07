from cd_ortho.core.data import InputDataKeys, InputDType, TargetTYPES

InputDataFields = {InputDataKeys.INPUT: {"name": "image", "type": "image", "dtype": InputDType.UINT8},
                   InputDataKeys.TARGET: {"name": "mask", "type": TargetTYPES.MASK, "encoding": "integer"}
                   }
