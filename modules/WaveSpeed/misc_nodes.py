import importlib
import json

from . import utils


class EnhancedCompileModel:

    def patch(
        self,
        model,
        is_patcher,
        object_to_patch,
        compiler,
        fullgraph,
        dynamic,
        mode,
        options,
        disable,
        backend,
    ):
        utils.patch_optimized_module()
        utils.patch_same_meta()

        import_path, function_name = compiler.rsplit(".", 1)
        module = importlib.import_module(import_path)
        compile_function = getattr(module, function_name)

        mode = mode if mode else None
        options = json.loads(options) if options else None

        if compiler == "torch.compile" and backend == "inductor" and dynamic:
            # TODO: Fix this
            # File "pytorch/torch/_inductor/fx_passes/post_grad.py", line 643, in same_meta
            #   and statically_known_true(sym_eq(val1.size(), val2.size()))
            #   AttributeError: 'SymInt' object has no attribute 'size'
            pass

        if is_patcher:
            patcher = model[0].clone()
        else:
            patcher = model.patcher
            patcher = patcher.clone()

        patcher.add_object_patch(
            object_to_patch,
            compile_function(
                patcher.get_model_object(object_to_patch),
                fullgraph=fullgraph,
                dynamic=dynamic,
                mode=mode,
                options=options,
                disable=disable,
                backend=backend,
            ),
        )

        if is_patcher:
            return (patcher,)
        else:
            model.patcher = patcher
            return (model,)
