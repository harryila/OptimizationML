FROM localhost:5000/p23-runtime@sha256:44ef23717780b1cbf112b183e7988b1319ddfed6b1d224efaa33e1e6d96de4c1
USER root

COPY prepare_p24_immutable_remote_module.py \
    /opt/p24-build/prepare_p24_immutable_remote_module.py

RUN /opt/p23-venv/bin/python \
        /opt/p24-build/prepare_p24_immutable_remote_module.py \
    && PYTHONDONTWRITEBYTECODE=1 /opt/p23-venv/bin/python -c \
        'import pathlib, sys, torch; parameter = torch.nn.Parameter(torch.zeros(1)); torch.optim.SGD([parameter], lr=1.0); expected = pathlib.Path("/opt/p23-venv/lib/python3.12/site-packages/torch/_p24_generated_remote_modules/_remote_module_non_scriptable.py"); observed = pathlib.Path(sys.modules["_remote_module_non_scriptable"].__file__); sys.exit("P24 generated module did not load from the exact image path") if observed.resolve() != expected else None' \
    && /opt/p23-venv/bin/python \
        /opt/p24-build/prepare_p24_immutable_remote_module.py --verify-only
