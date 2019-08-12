from HyperAPI.hdp_api.base.resource import Resource
from HyperAPI.hdp_api.base.route import Route


class Anaplan(Resource):
    name = "anaplan"
    available_since = "4.2.6"
    removed_since = None

    class _getWorkspaces(Route):
        name = "getWorkspaces"
        httpMethod = Route.GET
        path = "/anaplan/workspaces"

    class _getModels(Route):
        name = "getModels"
        httpMethod = Route.GET
        path = "/anaplan/models"

    class _getFiles(Route):
        name = "getFiles"
        httpMethod = Route.GET
        path = "/anaplan/workspaces/{workspace_id}/model/{model_id}"
        _path_keys = {
            'workspace_id': Route.VALIDATOR_OBJECTID,
            'model_id': Route.VALIDATOR_OBJECTID
        }
    class _getImports(Route):
        name = "getImports"
        httpMethod = Route.GET
        available_since = '4.2.8'
        path = "/anaplan/workspaces/{workspace_id}/model/{model_id}/imports/{dataset_Id}"
        _path_keys = {
            'workspace_id': Route.VALIDATOR_OBJECTID,
            'model_id': Route.VALIDATOR_OBJECTID,
            'dataset_Id': Route.VALIDATOR_OBJECTID
        }
    class _sendPayload(Route):
        name = "sendPayload",
        available_since = '4.2.8'
        httpMethod = Route.POST
        path = "/anaplan/send" 

    class _anaplanLogin(Route):
        name = "anaplanLogin"
        httpMethod = Route.POST
        path = "/anaplan/login"
