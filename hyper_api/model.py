from hypercube_api.util import Helper
from hypercube_api.hyper_api.base import Base
from hypercube_api.utils.exceptions import ApiException
from datetime import datetime
from pandas import read_csv
from numpy import reshape
from io import StringIO
from json import dump


class ModelFactory:
    """
    """
    _PURITY = 'Purity'
    _COVERAGE = 'Coverage'
    _LIFT = 'Lift'
    _HYPERCUBE_ALGO_TYPE = 'HyperCube'

    def __init__(self, api, project):
        self.__api = api
        self.__algo_list = ['HyperCube', 'LogisticRegression', 'DecisionTree', 'RandomForest', 'GradientBoosting']
        self.__dataset = project

    @Helper.try_catch
    def filter(self):
        """
        Get all models

        Returns:
            The list of models
        """
        project_id = self.__dataset.project_id

        data = {
            'projectId': project_id,
            'type': {'$in': ["prediction", "applyOtherModel", "applyHypercubeModel",
                             "predictionRuleset", "otherPrediction", "hypercubePrediction"]
                     }
        }
        json = {'project_ID': project_id, 'json': data}
        json_returned = self.__api.Task.task(**json)
        return [HyperCube(self.__api, model_json) if model_json.get('algoType') == self._HYPERCUBE_ALGO_TYPE
                else Model(self.__api, model_json) for model_json in json_returned]

    @Helper.try_catch
    def get(self, name):
        """
        Get a model matching the given name or None if there is no match

        Returns:
            The Model or None
        """
        models = list(filter(lambda x: x.name == name, self.filter()))
        if models:
            return models[0]
        return None

    @Helper.try_catch
    def get_by_id(self, id):
        """
        Get the model matching the given ID or None if there is no match

        Returns:
            The Model or None
        """
        models = [model for model in self.filter() if model.id == id]
        if models:
            return models[0]
        return None

    @Helper.try_catch
    def predict_from_ruleset(self, dataset_source, dataset_predict, rulesetname, name, target, nb_minimizations=1, coverage_increment=0.01):
        """
        Create a HyperCube classifier model from a ruleset

        Args:
            dataset_source (Dataset): Dataset where the ruleset was performed
            dataset_predict (Dataset): Dataset used to make these predictions
            rulesetname (str): Name of the ruleset you want to use to create the model
            name (str): Name of the new model
            target (Target): Target used to generate the model
            nb_minimizations (int): Number of minimizations to perform on the ruleset, default is 1
            coverage_increment (float): Percentage increment of target samples that a new rule must bring to be added to the minimized ruleset,
                default is 0.01

        Returns:
            the Model
        """
        kpiData = []
        for _id, _type in zip(target.score_ids, target.scores):
            kpiData.append({'kpiId': _id,
                            'type': _type,
                            'kpiFamily': target.indicator_family,
                            'scoreType': _type,
                            'kpiType': target.indicator_type,
                            'output': target.variable_name,
                            'kpiName': target.name,
                            'omodality': target.modality
                            })

        json = {'projectId': dataset_predict.project_id,
                'datasetId': dataset_predict.dataset_id,
                'datasetName': dataset_predict.name,
                'type': 'predictionRuleset',
                'params': {'modelName': name,
                           'sourceDatasetId': dataset_source.dataset_id,
                           'sourceFileName': dataset_predict.source_file_name,
                           'rulesetName': rulesetname,
                           'delimiter': dataset_source.delimiter,
                           'algoType': 'HyperCube',
                           'target': kpiData,
                           'nbMinimizations': nb_minimizations,
                           'coverageIncrement': coverage_increment
                           }
                }

        model = self.__api.Task.createtask(project_ID=self.__dataset.project_id, json=json)
        self._api.handle_work_states(self.project_id, work_type='predictionRuleset', work_id=model.get('_id'))
        return HyperCube(self.__api, model)

    @Helper.try_catch
    def create_hypercube(self, dataset, name, target, purity_min=None, coverage_min=None, rule_complexity=2, quantiles=10, min_marginal_contribution=None,
                         max_complexity=3, nb_minimizations=1, coverage_increment=0.01, split_ratio=0.7, nb_iterations=1,
                         purity_tolerance=0.1, enable_custom_discretizations=True, save_all_rules=False):
        """
        Create a HyperCube classifier model

        Args:
            dataset (Dataset): Dataset the model is fitted on
            name (str): Name of the new model
            target (Target): Target used to generate the model
            purity_min (float): Minimum purity of rules, default is the entire dataset purity
            coverage_min (int): Minimum coverage of the target population for each rule, default is 10
            rule_complexity (int): Maximum number of variables in rules, default is 2
            quantiles (int): Number of bins for all continuous numeric variables during quantization, default is 10
            min_marginal_contribution (float): a new rule R', created by adding a new constraint to an existing rule R (and thus increasing its complexity),
                is added to the ruleset if and only if it increases the original purity of R by the minimum marginal contribution or more. Default is 0.1
            max_complexity (int): maximum number of variables contained in the rules created during the local complexity increase phase. Default is 3
            nb_minimizations (int): Number of minimizations to perform on the ruleset, default is 1
            coverage_increment (float): Percentage increment of target samples that a new rule must bring to be added to the minimized ruleset,
                default is 0.01
            split_ratio (float): the first step in the model generation is the random split of the original dataset into a learning (or train) dataset
                representing by default 70% of the original dataset, and a validation (or test) dataset containing the remaining 30%. Default is 0.7
            nb_iterations (int): The final model is the result of several models based on different splits of the original dataset, using a bootstrap method.
                The parameter "Number of iterations" corresponds to the number of these splits that are made. Default is 1
            purity_tolerance (float): maximum spread between the purities of the rules applied to the learning and validation datasets
            enable_custom_discretizations (boolean): when ticked use the custom discretization(s) link to the selected dataset,
                eventually use "Quantiles" parameter for remaining variables. Default is True
            save_all_rules (boolean): save all generated rules in a new ruleset. Default is False

        Returns:
            the created model
        """
        variable = next(variable for variable in dataset.variables if variable.name == target.variable_name)
        index = variable.modalities.index(target.modality)
        datasetPurity = variable.purities[index]
        score_purity_min = purity_min or round(datasetPurity, 3)

        if min_marginal_contribution is None:
            if score_purity_min > 0.99:
                min_marginal_contribution = round(1 / score_purity_min - 1, 3)
            elif score_purity_min > 0.9:
                min_marginal_contribution = round(0.99 / score_purity_min - 1, 3)
            else:
                min_marginal_contribution = 0.1

        coverage_min = coverage_min or 10 if (variable.frequencies[index] < 1000) else 0.01

        if enable_custom_discretizations is True:
            discretizations = dataset._discretizations
        else:
            discretizations = {}

        scores = []
        for score_id, score_type in zip(target.score_ids, target.scores):
            score = {
                'deleted': False,
                'kpiFamily': target.indicator_family,
                'kpiName': target.name,
                'kpiType': target.indicator_type,
                'omodality': target.modality,
                'output': target.variable_name,
                'projectId': target.project_id,
                'scoreType': score_type,
                '_id': score_id
            }
            if score_type == self._PURITY:
                score['minValue'] = score_purity_min
            elif score_type == self._COVERAGE:
                score['minValue'] = coverage_min
            elif score_type == self._LIFT:
                score['minValue'] = 1
            scores.append(score)

        kpisel = {
            'datasetPurity': datasetPurity,
            'kpiFamily': target.indicator_family,
            'kpiName': target.name,
            'kpiType': target.indicator_type,
            'projectId': target.project_id,
            'scores': scores,
            'selectedBy': 'target'
        }

        params = {
            'algoType': 'HyperCube',
            'complexityExhaustive': rule_complexity,
            'countQuantiles': quantiles,
            'coverageIncrement': coverage_increment,
            'coverageThreshold': 10,
            'delimiter': 'semicolon',
            'discretizations': discretizations,
            'dtmaxdepth': 4,
            'elasticNetParam': 0,
            'enableCustomDiscretizations': enable_custom_discretizations,
            'featureSubsetStrategy': 'sqrt',
            'gbmaxdepth': 3,
            'gbn_estimators': 100,
            'kpis': [],
            'learning_rate': 0.1,
            'lrCost': 1,
            'maxComplexity': max_complexity,
            'maxDepth': 3,
            'maxIter': 100,
            'minInfoGain': 0,
            'minInstance': 1,
            'minMarginalContribution': min_marginal_contribution,
            'minObservation': 3,
            'missingValues': 0.1,
            'modelName': name,
            'nbMaxModality': 50,
            'nbMinObservation': 10,
            'nbMinimizations': nb_minimizations,
            'nbModels': nb_iterations,
            'numTrees': 10,
            'percentageSplit': split_ratio,
            'purityThreshold': score_purity_min,
            'purityTolerance': purity_tolerance,
            'regParam': 0,
            'replaceMissingValues': 'Median',
            'rfmaxdepth': 2,
            'rfn_estimators': 100,
            'saveAllRules': 1 if save_all_rules else 0,
            'sourceFileName': dataset.source_file_name,
            'splitRatio': split_ratio,
            'stepSize': 0.1,
            'subsamplingRate': 1,
            'target': [score for score in scores if score['scoreType'] == self._PURITY or score['scoreType'] == self._COVERAGE],
            'tol': 0.000001
        }

        data = {
            'algo': 'HyperCube',
            'algolist': ['HyperCube', 'LogisticRegression', 'DecisionTree', 'RandomForest', 'GradientBoosting'],
            'datasetId': dataset.dataset_id,
            'datasetName': dataset.name,
            'dtcr': 'gini',
            'enableCustomDiscretizations': enable_custom_discretizations,
            'gbloss': 'deviance',
            'keyIndicators': [],
            'kpisel': kpisel,
            'lrPenalty': 'l2',
            'lrSolver': 'liblinear',
            'modelFile': '',
            'modelName': name,
            'params': params,
            'projectId': dataset.project_id,
            'rfcr': 'gini',
            'saveAllRules': save_all_rules,
            'selectedDataset': dataset._json,
            'sourceFileName': '',
            'spark': False,
            'type': 'hypercubePrediction',
            'validTarget': True
        }

        json = {'project_ID': dataset.project_id, 'json': data}
        json_returned = self.__api.Task.createtask(**json)

        try:
            self.__api.handle_work_states(dataset.project_id, work_type=json_returned.get('type'), work_id=json_returned.get('_id'))
        except Exception as E:
            raise ApiException('Unable to create the HyperCube model ' + name, str(E))

        return HyperCube(self.__api, json_returned)

    @Helper.try_catch
    def get_or_create_hypercube(self, dataset, name, target=None, purity_min=None, coverage_min=None, rule_complexity=2, quantiles=10,
                                min_marginal_contribution=None, max_complexity=3, nb_minimizations=1, coverage_increment=0.01, split_ratio=0.7,
                                nb_iterations=1, purity_tolerance=0.1, enable_custom_discretizations=True, save_all_rules=False):
        """
        Get or Create a HyperCube classifier model

        Args:
            dataset (Dataset): Dataset the model is fitted on
            name (str): Name of the model
            target (Target): Target used to generate the model
            purity_min (float): Minimum purity of rules, default is the entire dataset purity
            coverage_min (int): Minimum coverage of the target population for each rule, default is 10
            rule_complexity (int): Maximum number of variables in rules, default is 2
            quantiles (int): Number of bins for all continuous numeric variables during quantization, default is 10
            min_marginal_contribution (float): a new rule R', created by adding a new constraint to an existing rule R (and thus increasing its complexity),
                is added to the ruleset if and only if it increases the original purity of R by the minimum marginal contribution or more. Default is 0.1
            max_complexity (int): maximum number of variables contained in the rules created during the local complexity increase phase. Default is 3
            nb_minimizations (int): Number of minimizations to perform on the ruleset, default is 1
            coverage_increment (float): Percentage increment of target samples that a new rule must bring to be added to the minimized ruleset,
                default is 0.01
            split_ratio (float): the first step in the model generation is the random split of the original dataset into a learning (or train) dataset
                representing by default 70% of the original dataset, and a validation (or test) dataset containing the remaining 30%. Default is 0.7
            nb_iterations (int): The final model is the result of several models based on different splits of the original dataset, using a bootstrap method.
                The parameter "Number of iterations" corresponds to the number of these splits that are made. Default is 1
            purity_tolerance (float): maximum spread between the purities of the rules applied to the learning and validation datasets
            enable_custom_discretizations (boolean): when ticked use the custom discretization(s) link to the selected dataset,
                eventually use "Quantiles" parameter for remaining variables. Default is True
            save_all_rules (boolean): save all generated rules in a new ruleset. Default is False

        Returns:
            the created model or existing model matching name parameter
        """

        model = self.get(name)
        if model is not None:
            return model

        return self.create_hypercube(dataset, name, target, purity_min, coverage_min, rule_complexity, quantiles, min_marginal_contribution,
                                     max_complexity, nb_minimizations, coverage_increment, split_ratio, nb_iterations, purity_tolerance,
                                     enable_custom_discretizations, save_all_rules)


class Model(Base):
    """
    """
    def __init__(self, api, json_return):
        self.__api = api
        self.__json_returned = json_return
        self._is_deleted = False
        self.__json_confusion_matrix = None

    def __repr__(self):
        return """\n{} : {} <{}>\n""".format(
            'Model',
            self.name,
            self.id
        ) + ("\t<! This model has been deleted>\n" if self._is_deleted else "") + \
            """\t- Algo : {}\n\t- Dataset : {}\n\t- Target : {}\n\t- Created on : {}\n""".format(
            self.algoType,
            self.dataset_name,
            self.kpi_name,
            self.created.strftime('%Y-%m-%d %H:%M:%S UTC'))

    @property
    def _json(self):
        return self.__json_returned

    @property
    def algoType(self):
        return self.__json_returned.get('algoType')

    @property
    def dataset_name(self):
        return self.__json_returned.get('datasetName')

    @property
    def id(self):
        return self.__json_returned.get('_id')

    @property
    def kpi_name(self):
        return self.__json_returned.get('kpiName')

    @property
    def project_id(self):
        return self.__json_returned.get('projectId')

    @property
    def name(self):
        return self.__json_returned.get('modelName')

    @property
    @Helper.try_catch
    def created(self):
        return self.str2date(self.__json_returned.get('createdAt'), '%Y-%m-%dT%H:%M:%S.%fZ')

    @Helper.try_catch
    def delete(self):
        if not self._is_deleted:
            json = {'project_ID': self.project_id, 'task_ID': self.id}
            self.__api.Task.deletetask(**json)
            self._is_deleted = True
        return self

    @property
    @Helper.try_catch
    def area_under_roc(self):
        self.__load_confusion_matrix()
        return self.__json_confusion_matrix['ROC curve'][-1]['auc']

    @Helper.try_catch
    def get_confusion_matrix(self, top_score_ratio):
        if not 0 <= top_score_ratio <= 1:
            raise ApiException('top_score_ratio must be greater or equal to 0 and lower or equal to 1')

        self.__load_confusion_matrix()
        index = self.__get_index(top_score_ratio)
        values = self.__json_confusion_matrix['Lift curve'][index]
        return ConfusionMatrix(true_positives=values['TP'], false_positives=values['FP'],
                               true_negatives=values['TN'], false_negatives=values['FN'])

    def __get_index(self, top_score_ratio):
        if top_score_ratio == 0:
            return 0
        else:
            length = len(self.__json_confusion_matrix['Lift curve'])
            return max(0, round(length * top_score_ratio) - 1)

    @Helper.try_catch
    def display_curve(self, curve='ROC curve', title=None, model_line=None, random_line=None, legend=None):
        """
        Plot the selected curve of this model

        Args:
            curve (str): curve to be diplayed, options are 'ROC curve', 'Gain curve', 'Lift curve', 'Purity curve' and
                'Precision Recall'. Default is 'ROC curve'.
            title (str): Title of the diagram. Default is a custom model name
            model_line (dict): display options of model line, ex: dict(color=('rgb(205, 12, 24)'), dash='dash', width=1).
                Default is a blue line. see https://plot.ly/python/line-and-scatter/
            random_line (dict): display options of random line. Default is a red dash line.
            legend (dict): legend options, ex: dict(orientation="h") or dict(x=-.1, y=1.2).
                Default is at the right of the diagram. see https://plot.ly/python/legend/
        """

        try:
            import plotly.graph_objs as go
            import plotly.offline as py
            from plotly.offline import init_notebook_mode
        except ImportError as E:
            raise ApiException('Plotly external package is required for this operation, please execute "!pip install plotly" and restart the kernel', str(E))

        if curve not in ['ROC curve', 'Gain curve', 'Lift curve', 'Purity curve', 'Precision Recall']:
            print('Unexpected curve type : {}, valid options are : {}'.format(curve,
                  "'ROC curve', 'Gain curve', 'Lift curve', 'Purity curve', 'Precision Recall'"))
            return

        self.__load_confusion_matrix()

        x, y, x_name, y_name = self.__get_x_y_info(curve)

        init_notebook_mode(connected=False)
        if model_line:
            roc = go.Scatter(x=x, y=y, name='{}'.format(self.name), mode='lines', line=model_line)
        else:
            roc = go.Scatter(x=x, y=y, name='{}'.format(self.name), mode='lines')

        data = [roc]
        random_line_arg = random_line or dict(color=('rgb(205, 12, 24)'), dash='dash', width=1)
        if curve == 'ROC curve' or curve == 'Gain curve':
            random = go.Scatter(x=[0, 1], y=[0, 1], name='Random', mode='lines', line=random_line_arg)
        elif curve == 'Lift curve':
            random = go.Scatter(x=[0, 1], y=[1, 1], name='Random', mode='lines', line=random_line_arg)
        else:
            random = None

        if random:
            data.append(random)

        default_title = '{} of {}'.format(curve, self.name)
        if curve == 'ROC curve' or curve == 'Precision Recall':
            default_title = '{} (AUC = {:0.2f})'.format(default_title,
                                                        self.__json_confusion_matrix[curve][-1]['auc'])
        curve_title = title or default_title

        layout = dict(title=curve_title, xaxis=dict(title=x_name, range=[0, 1]),
                      yaxis=dict(title=y_name, range=[0, max(y) + 0.05]),)
        if legend:
            layout['legend'] = legend

        fig = dict(data=data, layout=layout)
        py.iplot(fig, validate=False)

    def __get_x_y_info(self, curve):
        if curve == 'ROC curve':
            x = [point['FPR'] for point in self.__json_confusion_matrix[curve]]
            y = [point['Sensitivity'] for point in self.__json_confusion_matrix[curve]]
            x_name = 'False Positive Rate'
            y_name = 'True Positive Rate'
        elif curve == 'Gain curve':
            x = [point['TopScore'] for point in self.__json_confusion_matrix[curve]]
            y = [point['Sensitivity'] for point in self.__json_confusion_matrix[curve]]
            x_name = 'Top score percentages'
            y_name = 'Recall'
        elif curve == 'Lift curve':
            x = [point['TopScore'] for point in self.__json_confusion_matrix[curve]]
            y = [point['Lift'] for point in self.__json_confusion_matrix[curve]]
            x_name = 'Top score percentages'
            y_name = 'Target lift'
        elif curve == 'Purity curve':
            x = [point['TopScore'] for point in self.__json_confusion_matrix[curve]]
            y = [point['Purity'] for point in self.__json_confusion_matrix[curve]]
            x_name = 'Top score percentages'
            y_name = 'Precision'
        elif curve == 'Precision Recall':
            x = [point['Sensitivity'] for point in self.__json_confusion_matrix[curve]]
            y = [point['Purity'] for point in self.__json_confusion_matrix[curve]]
            x_name = 'Recall'
            y_name = 'Precision'
        return x, y, x_name, y_name

    def __load_confusion_matrix(self):
        if not self.__json_confusion_matrix:
            json = {'project_ID': self.project_id, 'model_ID': self.id, 'dataset_ID': self.__json_returned.get('datasetId')}
            self.__json_confusion_matrix = self.__api.Prediction.getconfusionmatrix(**json)


class HyperCube(Model):
    """
    """
    def __init__(self, api, json_return):
        self.__api = api
        self.__json_returned = json_return
        super().__init__(api, json_return)

    @Helper.try_catch
    def apply(self, dataset, applied_model_name, add_score_to_dataset=False, score_column_name=None):
        """
        Apply the HyperCube classifier model on a selected data set

        Args:
            dataset (Dataset): Dataset the model is applied on
            applied_model_name (str): Name of the new applied model
            add_score_to_dataset (boolean): if set to True a new column containing the scores is added to the dataset.
                Default is False.
            score_column_name (str): name of the score column, used only if add_score_to_dataset is set to True
        Returns:
            the applied Model
        """

        params = dict(self.__json_returned)
        params['modelName'] = applied_model_name

        data = {
            'datasetId': dataset.dataset_id,
            'datasetName': dataset.name,
            'fromDatasetId': self.__json_returned.get('datasetId'),
            'modelId': self.__json_returned.get('_id'),
            'params': params,
            'projectId': dataset.project_id,
            'spark': False,
            'type': 'applyPrediction'
        }

        if add_score_to_dataset:
            data['saveScore'] = score_column_name or 'score_' + applied_model_name

        json = {'project_ID': dataset.project_id, 'json': data}
        json_returned = self.__api.Task.createtask(**json)
        try:
            self.__api.handle_work_states(dataset.project_id, work_type=json_returned.get('type'), work_id=json_returned.get('_id'))
        except Exception as E:
            raise ApiException('Unable to create the applied HyperCube model ' + applied_model_name, str(E))

        return HyperCube(self.__api, json_returned)

    @Helper.try_catch
    def export_scores(self, path, variables=None):
        """
        Export the scores of this model in a csv file

        Args:
            path (str): the destination path for the exported scores
            variables (list of Variable): the variables of the dataset to add in the file. Default is None
        """
        data = {
            'datasetId': self.__json_returned.get('datasetId'),
            'columns': [variable.name for variable in variables] if variables else []
        }
        json = {'project_ID': self.project_id, 'model_ID': self.id, 'json': data}
        json_returned = self.__api.Prediction.postexportscores(**json)

        try:
            self.__api.handle_work_states(self.project_id, work_type=json_returned.get('type'), work_id=json_returned.get('_id'))
        except Exception as E:
            raise ApiException('Unable to export the model scores for ' + self.name, str(E))

        outputFile = json_returned.get('workParams').get('outputFile').split("_")[1]
        data = {
            'outputFile': outputFile
        }
        json = {'project_ID': self.project_id, 'model_ID': self.id, 'params': data}
        to_export = self.__api.Prediction.getexportscores(**json)

        with open(path, 'wb') as FILE_OUT:
            FILE_OUT.write(to_export)

    @Helper.try_catch
    def predict_scores(self, dataset, keep_applied_model=False):
        """
        Predict target scores for input dataset

        Args:
            dataset (Dataset): the dataset containing the input samples.
            keep_applied_model (boolean): A HyperCube applied model is temporarily created to compute these scores,
                set this parameter to True if you want this model to be persisted. Default is False.

        Returns:
            a NumPy array of shape [n_samples,] where n_samples is the number of samples in the input dataset
        """
        applied_model = self.apply(dataset, '{}_applied_{}'.format(self.name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        data = {
            'datasetId': dataset.dataset_id,
            'columns': []
        }
        json = {'project_ID': applied_model.project_id, 'model_ID': applied_model.id, 'json': data}
        json_returned = self.__api.Prediction.postexportscores(**json)

        try:
            self.__api.handle_work_states(applied_model.project_id, work_type=json_returned.get('type'), work_id=json_returned.get('_id'))
        except Exception as E:
            raise ApiException('Unable to get the model scores for {}'.format(self.name), str(E))

        outputFile = json_returned.get('workParams').get('outputFile').split("_")[1]
        data = {
            'outputFile': outputFile
        }
        json = {'project_ID': applied_model.project_id, 'model_ID': applied_model.id, 'params': data}
        scores = self.__api.Prediction.getexportscores(**json)
        scoreIO = StringIO(scores.decode('utf-8'))

        try:
            df = read_csv(scoreIO, sep=';', skiprows=1, usecols=[1])
        except Exception as E:
            raise ApiException('Unable to read the model scores for {}'.format(self.name), str(E))

        if not keep_applied_model:
            applied_model.delete()

        return reshape(df.values, (df.values.shape[0]))

    @Helper.try_catch
    def export_model(self, path, format='Python'):
        """
        Export this model in a local file

        Args:
            path (str): the destination path for the exported model
            format (str): {'Python', 'csv', 'JSON', 'R', 'Scala', 'Java', 'JavaScript', 'MySQL', 'PLSQL', 'TSQL'}. Default is 'Python'
        """
        URL_PREFIX = 'api/v1/'
        data = {
            'format': format
        }
        json = {'project_ID': self.project_id, 'model_ID': self.id, 'params': data}
        url = self.__api.Prediction.exportrules(**json)
        url = url.decode("utf-8").replace(URL_PREFIX, '')
        to_export = self.__api.session.request(url=url, method='get')
        if isinstance(to_export, bytes):
            with open(path, 'wb') as FILE_OUT:
                FILE_OUT.write(to_export)
        else:
            with open(path, 'w') as FILE_OUT:
                dump(to_export, FILE_OUT)


class ConfusionMatrix:

    def __init__(self, true_positives, false_positives, true_negatives, false_negatives):
        self.true_positives = true_positives
        self.false_positives = false_positives
        self.true_negatives = true_negatives
        self.false_negatives = false_negatives
