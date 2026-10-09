def select_projectors(pipeline, requested):
    """Keep requested projectors and only the connectors they use."""
    if requested is not None:
        for proj_name in list(pipeline.projectors.keys()):
            if proj_name not in requested:
                assert pipeline.remove_projector(proj_name)

        for conn_name in list(pipeline.connectors.keys()):
            if not any(proj_name.startswith(conn_name) for proj_name in requested):
                assert pipeline.remove_connector(conn_name)

    return list(pipeline.projectors.keys())
