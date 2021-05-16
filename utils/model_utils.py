"""
Utility functions related to modifying torch model properties.
"""


def freeze_layers(model, frozen_dict):
    for c in model.named_children():
        name, child = c[0], c[1]
        if name in frozen_dict and frozen_dict[name] != '*':
            frozen_list = frozen_dict[name]
            for p, v in child.named_parameters():
                freeze = True
                for f_el in frozen_list:
                    if p.startswith(f_el):
                        freeze = False
                        break
                if freeze:
                    v.requires_grad = False
                    print(f"freezing layer {p}, frozen_dict={frozen_dict}")
            freeze_layers(child, frozen_dict)


def unfreeze_layers(model):
    for c in model.named_children():
        for p, v in c[1].named_parameters():
            v.requires_grad = True
        unfreeze_layers(c[1])
