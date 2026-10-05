"""ArXiv AI assistant backend.

Layering (import direction is strictly top-down):

    api / cli  ->  services  ->  pipeline steps
                        |            |
                        v            v
              clients / embeddings / db.vector_store
                        |
                        v
                domain (pure)  +  infra (transport)
"""

__version__ = "0.1.0"