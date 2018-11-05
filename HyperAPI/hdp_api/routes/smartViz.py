from HyperAPI.hdp_api.routes import Resource, Route
from HyperAPI.hdp_api.routes.base.version_management import reroute


class _getSmartDataVizsWithoutDataset(Route):
    name = "getSmartDataVizs"
    httpMethod = Route.GET
    path = "/projects/{project_ID}/smartDataViz/"
    _path_keys = {
        'project_ID': Route.VALIDATOR_OBJECTID,
    }

    @staticmethod
    def convert_path(**kwargs):
        kwargs.pop('dataset_ID')
        return kwargs


class SmartDataViz(Resource):
    name = "SmartDataViz"

    class _getSmartDataViz(Route):
        name = "getSmartDataViz"
        httpMethod = Route.GET
        path = "/projects/{project_ID}/datasets/{dataset_ID}/smartDataViz/{smartDataViz_ID}"
        _path_keys = {
            'project_ID': Route.VALIDATOR_OBJECTID,
            'dataset_ID': Route.VALIDATOR_OBJECTID,
            'smartDataViz_ID': Route.VALIDATOR_OBJECTID,
        }

    class _computeSmartDataViz(Route):
        name = "computeSmartDataViz"
        httpMethod = Route.POST
        path = "/projects/{project_ID}/datasets/{dataset_ID}/smartDataViz/"
        _path_keys = {
            'project_ID': Route.VALIDATOR_OBJECTID,
            'dataset_ID': Route.VALIDATOR_OBJECTID,
        }

    @reroute("3.0", _getSmartDataVizsWithoutDataset, _getSmartDataVizsWithoutDataset.convert_path)
    class _getSmartDataVizs(Route):
        name = "getSmartDataVizs"
        httpMethod = Route.GET
        path = "/projects/{project_ID}/datasets/{dataset_ID}/smartDataViz/"
        _path_keys = {
            'project_ID': Route.VALIDATOR_OBJECTID,
            'dataset_ID': Route.VALIDATOR_OBJECTID,
        }

    class _deleteSmartDataViz(Route):
        name = "deleteSmartDataViz"
        httpMethod = Route.POST
        path = "/projects/{project_ID}/datasets/{dataset_ID}/smartDataViz/{smartDataViz_ID}/delete"
        _path_keys = {
            'project_ID': Route.VALIDATOR_OBJECTID,
            'dataset_ID': Route.VALIDATOR_OBJECTID,
            'smartDataViz_ID': Route.VALIDATOR_OBJECTID,
        }
