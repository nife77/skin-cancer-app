for layer in model.layers:
    if "out" in layer.name:
        print(layer.name)
