"""TAMBO detector-layout optimization library.

Six domain subpackages, each re-exporting its public names:

    geometry    the mountain mesh, its surface, and detector placement on it
    layouts     layout generators and the learnable (x, y) parameterization
    showers     shower corpora and the ground-truth response kernel
    surrogates  the DeepSets surrogates and the reconstruction net
    data        Step-1 dataset construction
    optimize    the U(x, y) objective and its terms

Importing this package has no side effects — it does not touch `sys.path`. One
module does, deliberately: `showers.generate` needs the sibling repo at
`TAMBO-opt` and is therefore not re-exported, so `import modules.showers` stays
safe where that repo is absent. Import it by full path when you need it.
"""
