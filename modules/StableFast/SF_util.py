def hash_arg(arg: any) -> any:
    """#### Hash an argument.

    #### Args:
        - `arg` (any): The argument to hash.

    #### Returns:
        - `any`: The hashed argument.
    """
    if isinstance(arg, (str, int, float, bytes)):
        return arg
    if isinstance(arg, (tuple, list)):
        return tuple(map(hash_arg, arg))
    if isinstance(arg, dict):
        return tuple(
            sorted(
                ((hash_arg(k), hash_arg(v)) for k, v in arg.items()), key=lambda x: x[0]
            )
        )
    return type(arg)