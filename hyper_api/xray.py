from hypercube_api.util import Helper
from hypercube_api.hyper_api.base import Base
from hypercube_api.utils.exceptions import ApiException
from hypercube_api.hyper_api.xrayvariable import XRayVariableFactory
from hypercube_api.hyper_api.target import TargetFactory, Target, Description


class XrayFactory:
    """
    """
    def __init__(self, api, project_id):
        self.__api = api
        self.__project_id = project_id

    @Helper.try_catch
    def create(self, dataset, name, target,
               quantiles=10, enable_custom_discretizations=True):
        """
        Args:
            dataset (Dataset) : dataset on which Xray will be created
            name (str): name of the Xray to create
            target (Target or Description): target to generate the Xray
            quantiles (int): Number of intervals the continuous variables are quantized in, default is 10
            enable_custom_discretizations (boolean): use custom discretizations, eventually use "quantiles" parameter for remaining variables, default is True

        Returns:
            Xray
        """
        project_id = self.__project_id
        dataset_id = dataset.dataset_id
        dataset_name = dataset.name
        dataset_sourceFileName = dataset.source_file_name
        dataset_separator = dataset.separator

        if enable_custom_discretizations is True:
            discretizations = dataset._discretizations
        else:
            discretizations = {}

        kpi_data = {
            "kpiName": target.name,
            "kpiType": target.indicator_type,
            "output": target.variable_name,
            "kpiFamily": target.indicator_family,
            "scoreType": "Shift" if target.indicator_type == TargetFactory.KPI_TYPE_CONTINUOUS else "Lift",
            "isMainKey": False,
        }

        if isinstance(target, Target):
            dictionary = dict(zip(target.scores, target.score_ids))
            if target.indicator_type == TargetFactory.KPI_TYPE_DISCRETE or target.indicator_type == TargetFactory.KPI_TYPE_DISCRETE_MODALITY:
                kpi_data['_id'] = dictionary[TargetFactory.KPI_SCORE_PURITY]
                kpi_data["omodality"] = target.modality
            elif target.indicator_type == TargetFactory.KPI_TYPE_CONTINUOUS:
                kpi_data['_id'] = dictionary[TargetFactory.KPI_SCORE_AVERAGE_VALUE]
            else:
                raise ApiException('Unexpected target indicator type')

        elif isinstance(target, Description):
            kpi_data['_id'] = target.score_id
        else:
            raise ApiException('Unexpected target Structure')

        data = {
            "projectId": project_id,
            "task": {
                "type": "simplelift",
                "datasetName": dataset_name,
                "datasetId": dataset_id,
                "projectId": project_id,
                "params": {
                    "source": dataset_sourceFileName,
                    "kpis": [kpi_data],
                    "name": name,
                    "quantileOrder": quantiles,
                    "separator": dataset_separator,
                    "discretizations": discretizations
                }
            }
        }
        creation_json = self.__api.SimpleLift.newsimplelift(project_ID=self.__project_id, json=data)

        try:
            self.__api.handle_work_states(project_id, work_type='simplelift', work_id=creation_json.get('_id'))
        except Exception as E:
            raise ApiException('Unable to get the X-ray status', str(E))

        return Xray(self.__api, creation_json)

    @Helper.try_catch
    def filter(self):
        """
        Returns:
            list of Xrays on the current dataset
        """
        json = {'project_ID': self.__project_id}
        return [Xray(self.__api, x) for x in self.__api.SimpleLift.getsimplelifts(**json)]

    @Helper.try_catch
    def get(self, name):
        """
        Returns:
            Xray found by name
        """
        xrays = list(filter(lambda x: x.name == name, self.filter()))
        if xrays:
            return xrays[0]
        return None

    @Helper.try_catch
    def get_by_id(self, id):
        """
        Rreturns:
            Xray found by id
        """
        xrays = list(filter(lambda x: x.id == id, self.filter()))
        if xrays:
            return xrays[0]
        return None

    @Helper.try_catch
    def get_or_create(self, dataset, name, target, quantiles=10, enable_custom_discretizations=True):
        """
        find a Xray by name, or create it if not found

        Args:
            dataset (Dataset) : dataset on which Xray will be created
            name (str): name of the Xray to get or create
            target (Target or Description): target to generate the Xray
            quantiles (int): Number of intervals the continuous variables are quantized in, default is 10
            enable_custom_discretizations (boolean): use custom discretizations, eventually use "quantiles" parameter for remaining variables, default is True

        Returns:
            found or created Xray (Xray)
        """

        for xray in dataset.xrays:
            if (xray.name == name) and (xray.dataset_id == dataset.dataset_id):
                return xray

        return self.create(dataset, name, target, quantiles, enable_custom_discretizations)


class Xray(Base):
    """
    """
    def __init__(self, api, json_return):
        self.__api = api
        self.__json_returned = json_return
        self._is_deleted = False

    def __repr__(self):
        return """\n{} : {} <{}>\n""" \
            .format(self.__class__.__name__,
                    self.name,
                    self.id
                    ) + ("\t<! This Xray has been deleted>\n" if self._is_deleted else "") + \
            """\t- Dataset name : {}\n\t- Quantiles : {}\n\t- Created on : {}\n""".format(
                self.dataset_name,
                self.quantiles,
                self.created.strftime('%Y-%m-%d %H:%M:%S UTC'))

    @property
    def Variable(self):
        """
        Returns:
            factory to interact with Xray results (XRayVariableFactory)
        """
        if not self._is_deleted:
            return XRayVariableFactory(self.__api, self)

    @property
    def _json(self):
        return self.__json_returned

    @property
    def discretizations(self):
        return self.__json_returned.get('discretizations')

    @property
    def name(self):
        return self.__json_returned.get('name')

    @property
    def project_id(self):
        return self.__json_returned.get('projectId')

    @property
    def dataset_id(self):
        return self.__json_returned.get('datasetId')

    @property
    def dataset_name(self):
        return self.__json_returned.get('datasetName')

    @property
    def quantiles(self):
        return self.__json_returned.get('quantiles')

    @property
    def id(self):
        return self.__json_returned.get('_id')

    @property
    def variables(self):
        if not self._is_deleted:
            return self.Variable.sort()

    @property
    def created(self):
        return self.str2date(self.__json_returned.get('createdAt'), '%Y-%m-%dT%H:%M:%S.%fZ')

    @Helper.try_catch
    def delete(self):
        """
        Delete this X-ray
        """
        if not self._is_deleted:
            json = {'project_ID': self.project_id, 'task_ID': self.id}
            self.__api.Task.deletetask(**json)
            self._is_deleted = True
        return self
