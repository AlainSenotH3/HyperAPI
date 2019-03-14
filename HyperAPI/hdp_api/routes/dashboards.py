from HyperAPI.hdp_api.routes import Resource, Route


class Dashboards(Resource):
    name = "Dashboards"
    available_since = "3.0"
    removed_since = None

    class _Dashboards(Route):
        name = "getDashboards"
        httpMethod = Route.GET
        path = "/projects/{project_ID}/datasets/{dataset_ID}/dashboards"
        _path_keys = {
            'project_ID': Route.VALIDATOR_OBJECTID,
            'dataset_ID': Route.VALIDATOR_OBJECTID
        }

    class _ProjectDashboards(Route):
        name = "getProjectDashboards"
        available_since = '3.1'
        httpMethod = Route.GET
        path = "/projects/{project_ID}/dashboards"
        _path_keys = {
            'project_ID': Route.VALIDATOR_OBJECTID
        }

    class _GetDashboard(Route):
        name = "getDashboard"
        httpMethod = Route.GET
        path = "/projects/{project_ID}/datasets/{dataset_ID}/dashboards/{dashboard_ID}"
        _path_keys = {
            'project_ID': Route.VALIDATOR_OBJECTID,
            'dataset_ID': Route.VALIDATOR_OBJECTID,
            'dashboard_ID': Route.VALIDATOR_OBJECTID
        }

    class _GetProjectDashboard(Route):
        name = "getProjectDashboard"
        available_since = '3.1'
        httpMethod = Route.GET
        path = "/projects/{project_ID}/dashboards/{dashboard_ID}"
        _path_keys = {
            'project_ID': Route.VALIDATOR_OBJECTID,
            'dashboard_ID': Route.VALIDATOR_OBJECTID
        }

    class _addDashboard(Route):
        name = "addDashboard"
        httpMethod = Route.POST
        path = "/projects/{project_ID}/datasets/{dataset_ID}/dashboards"
        _path_keys = {
            'project_ID': Route.VALIDATOR_OBJECTID,
            'dataset_ID': Route.VALIDATOR_OBJECTID
        }

    class _addProjectDashboard(Route):
        name = "addProjectDashboard"
        available_since = '3.1'
        httpMethod = Route.POST
        path = "/projects/{project_ID}/dashboards"
        _path_keys = {
            'project_ID': Route.VALIDATOR_OBJECTID
        }

    class _updateDashboard(Route):
        name = "updateDashboard"
        available_since = '3.1'
        httpMethod = Route.POST
        path = "/projects/{project_ID}/datasets/{dataset_ID}/dashboards/{dashboard_ID}"
        _path_keys = {
            'project_ID': Route.VALIDATOR_OBJECTID,
            'dataset_ID': Route.VALIDATOR_OBJECTID,
            'dashboard_ID': Route.VALIDATOR_OBJECTID
        }

    class _updateProjectDashboard(Route):
        name = "updateProjectDashboard"
        available_since = '3.1'
        httpMethod = Route.POST
        path = "/projects/{project_ID}/dashboards/{dashboard_ID}"
        _path_keys = {
            'project_ID': Route.VALIDATOR_OBJECTID,
            'dashboard_ID': Route.VALIDATOR_OBJECTID
        }

    class _deleteDashboard(Route):
        name = "deleteDashboard"
        httpMethod = Route.POST
        path = "/projects/{project_ID}/datasets/{dataset_ID}/dashboards/{dashboard_ID}/delete"
        _path_keys = {
            'project_ID': Route.VALIDATOR_OBJECTID,
            'dataset_ID': Route.VALIDATOR_OBJECTID,
            'dashboard_ID': Route.VALIDATOR_OBJECTID
        }

    class _deleteProjectDashboard(Route):
        name = "deleteProjectDashboard"
        available_since = '3.1'
        httpMethod = Route.POST
        path = "/projects/{project_ID}/dashboards/{dashboard_ID}/delete"
        _path_keys = {
            'project_ID': Route.VALIDATOR_OBJECTID,
            'dashboard_ID': Route.VALIDATOR_OBJECTID
        }
