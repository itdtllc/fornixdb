"""Optional ecosystem adapters.

The fornixdb core is vendor-neutral and assumes nothing about any AI vendor's
tooling. Adapters import data FROM specific ecosystems into the store (one-way,
read-only on the source). A deployment that doesn't use that ecosystem simply
never invokes the adapter.
"""
