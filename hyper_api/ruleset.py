from hypercube_api.util import Helper
from hypercube_api.hyper_api.target import Description
from hypercube_api.hyper_api.base import Base
from hypercube_api.hyper_api.model import ModelFactory
from hypercube_api.hyper_api.rule import Rules, decode_kpiname_to_id
from hypercube_api.utils.exceptions import ApiException

import urllib.parse


class KeyIndicatorOption:
    """Creation of a key indicator"""
    def __init__(self, target, purity_min=None, purity_max=None, coverage_min=None, coverage_max=None,
                 lift_min=None, lift_max=None, zscore_min=None, zscore_max=None,
                 average_value_min=None, average_value_max=None, standard_deviation_min=None, standard_deviation_max=None,
                 shift_min=None, shift_max=None):
        self.target = target
        self.purity_min = purity_min
        self.purity_max = purity_max
        self.coverage_min = coverage_min
        self.coverage_max = coverage_max
        self.lift_min = lift_min
        self.lift_max = lift_max
        self.zscore_min = zscore_min
        self.zscore_max = zscore_max
        self.average_value_min = average_value_min
        self.average_value_max = average_value_max
        self.standard_deviation_min = standard_deviation_min
        self.standard_deviation_max = standard_deviation_max
        self.shift_min = shift_min
        self.shift_max = shift_max

    def __repr__(self):
        return "\n{} :".format(self.__class__.__name__) + \
               "{}".format(self.target) + \
               ("\t- Purity min: {}\n".format(self.purity_min) if self.purity_min is not None else "") + \
               ("\t- Purity max: {}\n".format(self.purity_max) if self.purity_max is not None else "") + \
               ("\t- Coverage min: {}\n".format(self.coverage_min) if self.coverage_min is not None else "") + \
               ("\t- Coverage max: {}\n".format(self.coverage_max) if self.coverage_max is not None else "") + \
               ("\t- Lift min: {}\n".format(self.lift_min) if self.lift_min is not None else "") + \
               ("\t- Lift max: {}\n".format(self.lift_max) if self.lift_max is not None else "") + \
               ("\t- Z-score min: {}\n".format(self.zscore_min) if self.zscore_min is not None else "") + \
               ("\t- Z-score max: {}\n".format(self.zscore_max) if self.zscore_max is not None else "") + \
               ("\t- Average value min: {}\n".format(self.average_value_min) if self.average_value_min is not None else "") + \
               ("\t- Average value max: {}\n".format(self.average_value_max) if self.average_value_max is not None else "") + \
               ("\t- Standard deviation min: {}\n".format(self.standard_deviation_min) if self.standard_deviation_min is not None else "") + \
               ("\t- Standard deviation max: {}\n".format(self.standard_deviation_max) if self.standard_deviation_max is not None else "") + \
               ("\t- Shift min: {}\n".format(self.shift_min) if self.shift_min is not None else "") + \
               ("\t- Shift max: {}\n".format(self.shift_max) if self.shift_max is not None else "")


class RulesetFactory:
    _PURITY = 'Purity'
    _COVERAGE = 'Coverage'
    _LIFT = 'Lift'
    _ZSCORE = 'Z-score'
    _AVERAGE_VALUE = 'Average value'
    _STANDARD_DEVIATION = 'Standard deviation'
    _SHIFT = 'Shift'
    _DISCRETE_MODALITY = "Discrete variable with a modality"
    _DISCRETE = "Discrete variable"
    _CONTINUOUS = "Continuous variable"

    def __init__(self, api, project_id):
        self.__api = api
        self.__project_id = project_id

    @Helper.try_catch
    def create_kpi_option(self, target, purity_min=None, purity_max=None, coverage_min=None, coverage_max=None,
                          lift_min=None, lift_max=None, zscore_min=None, zscore_max=None,
                          average_value_min=None, average_value_max=None, standard_deviation_min=None, standard_deviation_max=None,
                          shift_min=None, shift_max=None):
        """
        Create an additional key indicator
        Args:
            target (Target): Target to generate the key indicator
            purity_min (float):  Minimum value for purity score
            purity_max (float):  Maximum value for purity score
            coverage_min (float):  Minimum value for coverage score
            coverage_max (float):  Maximum value for coverage score
            lift_min (float):  Maximum value for lift score
            lift_max (float):  Maximum value for lift score
            zscore_min (float):  Minimum value for zscore score
            zscore_max (float):  Maximum value for zscore score
            average_value_min (float):  Minimum value for average value score
            average_value_max (float):  Maximum value for average value score
            standard_deviation_min (float):  Minimum value for standard deviation score
            standard_deviation_max (float):  Maximum value for standard deviation score
            shift_min (float):  Minimum value for shift score
            shift_max (float):  Maximum value for shift score
        Returns:
            key indicator
        """
        return KeyIndicatorOption(target=target, purity_min=purity_min, coverage_min=coverage_min, coverage_max=coverage_max,
                                  lift_min=lift_min, lift_max=lift_max, zscore_min=zscore_min, zscore_max=zscore_max,
                                  average_value_min=average_value_min, average_value_max=average_value_max,
                                  standard_deviation_min=standard_deviation_min, standard_deviation_max=standard_deviation_max,
                                  shift_min=shift_min, shift_max=shift_max)

    @Helper.try_catch
    def create(self, dataset, name, target, purity_min=None, coverage_min=None, lift_min=None, zscore_min=None, average_value_min=None,
               standard_deviation_max=None, shift_min=None, rule_complexity=2, quantiles=10,
               enable_custom_discretizations=True, min_marginal_contribution=None, compute_other_key_indicators=None,
               locally_increase_complexity=False, max_complexity=3, nb_minimizations=1, coverage_increment=0.01,
               validate_stability=False, split_ratio=0.7, nb_iterations=1, purity_tolerance=0.1):
        """
        Create a new ruleset

        Args:
            dataset (Dataset): Dataset used to generate the ruleset
            name (str): Name of the new ruleset
            target (Target): Target to generate the ruleset
            purity_min (float): Minimum purity of rules, default is the entire dataset purity (discrete target only)
            coverage_min (int): Minimum coverage of the target population for each rule, default is 10 (discrete target only)
            lift_min (float): Minimum lift, default is 1 (discrete target only)
            zscore_min (float): Minimum Z-score, default is None (discrete target only)
            average_value_min (float): Minimum average value, default is average value of the target on the whole dataset (continuous target only)
            standard_deviation_max (float) : Maximum standard deviation, default is None (continuous target only)
            shift_min (float): Minimum shift, default is None (continuous target only)
            rule_complexity (int): Maximum number of variables in rules, default is 2
            quantiles (int): Number of intervals the continuous variables are quantized in, default is 10
            enable_custom_discretizations (boolean): use custom discretizations, eventually use "quantiles" parameter for remaining variables, default is True
            min_marginal_contribution (float): a new rule R', created by adding a new constraint to an existing rule R (and thus increasing its complexity),
                is added to the ruleset if and only if it increases the original purity of R by the minimum marginal contribution or more. Default is 0.1
            compute_other_key_indicators (list of KeyIndicatorOption): Compute other Key Indicators.
            locally_increase_complexity (bool): Enable the locally increase complexity when set as true.
            max_complexity (int): Maximum numbers of features per rule.
            nb_minimizations (int):Interate the minimization process.
            coverage_increment (float): Percentage increment of target samples that a new rule must bring to be added to the minimization ruleset.
            validate_stability (bool): Enable to split your dataset, add iteration and set a purity tolerance when set as true.
            split_ratio (float): The percentage for the split (Between 0 and 1).
            nb_iterations (int): Number of iterations wanted.
            purity_tolerance (float): Purity tolerence allowed (Between 0 and 1).

        Returns:
            Ruleset
        """
        variable = next(variable for variable in dataset.variables if variable.name == target.variable_name)
        score_purity_min = None
        if (variable.is_discrete):
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
        else:
            min_marginal_contribution = 0.1

        if enable_custom_discretizations is True:
            discretizations = dataset._discretizations
        else:
            discretizations = {}

        if not compute_other_key_indicators:
            compute_other_key_indicators = []

        if not target:
            raise ApiException('You need a target to create a ruleset')
        if isinstance(target, Description):
            raise ApiException('Cannot perform a ruleset with a description kpi')

        data = {
            "projectId": self.__project_id,
            "task": {
                "type": "learning",
                "datasetId": dataset.dataset_id,
                "projectId": self.__project_id,
                "params": {
                    "learningName": name,
                    "datasetName": dataset.name,
                    "buildPredictiveModel": 0,
                    "sourceFileName": dataset.source_file_name,
                    "delimiter": dataset.separator,
                    "complexityExhaustive": rule_complexity,
                    "countQuantiles": quantiles,
                    "discretizations": discretizations,
                    "minMarginalContribution": min_marginal_contribution,
                    "target": [],
                    "kpis": []
                }
            }
        }

        for _id, _type in zip(target.score_ids, target.scores):
            _kpiData = {
                "kpiId": _id,
                "type": _type,
                "kpiFamily": target.indicator_family,
                "scoreType": _type,
                "kpiType": target.indicator_type,
                "output": target.variable_name,
                "kpiName": target.name,
                "omodality": target.modality
            }
            if _type == self._PURITY and score_purity_min is not None:
                _kpiData['minValue'] = score_purity_min
            elif _type == self._COVERAGE and coverage_min is not None:
                _kpiData['minValue'] = coverage_min
            elif _type == self._LIFT and lift_min is not None:
                _kpiData['minValue'] = lift_min
            elif _type == self._ZSCORE and zscore_min is not None:
                _kpiData['minValue'] = zscore_min
            elif _type == self._AVERAGE_VALUE and average_value_min is not None:
                _kpiData['minValue'] = average_value_min
            elif _type == self._STANDARD_DEVIATION and standard_deviation_max is not None:
                _kpiData['maxValue'] = standard_deviation_max
            elif _type == self._SHIFT and shift_min is not None:
                _kpiData['minValue'] = shift_min
            data['task']['params']['target'].append(_kpiData)

        msg = "Ruleset settings: \n\t- Target: {}".format(target.name) + \
              ("\n\t- Min Purity: {}".format(score_purity_min) if score_purity_min is not None else "") + \
              ("\n\t- Min Coverage: {}".format(coverage_min) if coverage_min is not None else "") + \
              ("\n\t- Min Lift: {}".format(lift_min) if lift_min is not None else "") + \
              ("\n\t- Min Z-score: {}".format(zscore_min) if zscore_min is not None else "") + \
              ("\n\t- Min Average value: {}".format(average_value_min) if average_value_min is not None else "") + \
              ("\n\t- Max Standard deviation: {}".format(standard_deviation_max) if standard_deviation_max is not None else "") + \
              ("\n\t- Min Shift: {}".format(shift_min) if shift_min is not None else "") + \
              "\n\t- Rule Complexity: {}\n\t- Default Number of Bins: {} \n\t- Enable custom discretizations: {}  \n\t- Min Marginal contribution: \
{}".format(rule_complexity, quantiles, enable_custom_discretizations, min_marginal_contribution)

        if (len(compute_other_key_indicators) > 0):
            for key_indicator in compute_other_key_indicators:
                for _id, _type in zip(key_indicator.target.score_ids, key_indicator.target.scores):
                    _kpiKI = {
                        "kpiId": _id,
                        "type": _type,
                        "kpiFamily": key_indicator.target.indicator_family,
                        "scoreType": _type,
                        "kpiType": key_indicator.target.indicator_type,
                        "output": key_indicator.target.variable_name,
                        "kpiName": key_indicator.target.name,
                        "omodality": key_indicator.target.modality
                    }
                    if (key_indicator.target.indicator_type == self._DISCRETE_MODALITY or key_indicator.target.indicator_type == self._DISCRETE):
                        if _type == self._PURITY:
                            if key_indicator.purity_min is not None:
                                _kpiKI['minValue'] = key_indicator.purity_min
                            if key_indicator.purity_max is not None:
                                _kpiKI['maxValue'] = key_indicator.purity_max
                        elif _type == self._COVERAGE:
                            if key_indicator.coverage_min is not None:
                                _kpiKI['minValue'] = key_indicator.coverage_min
                            if key_indicator.coverage_max is not None:
                                _kpiKI['maxValue'] = key_indicator.coverage_max
                        elif _type == self._LIFT:
                            if key_indicator.lift_min is not None:
                                _kpiKI['minValue'] = key_indicator.lift_min
                            if key_indicator.lift_max is not None:
                                _kpiKI['maxValue'] = key_indicator.lift_max
                        elif _type == self._ZSCORE:
                            if key_indicator.zscore_min is not None:
                                _kpiKI['minValue'] = key_indicator.zscore_min
                            if key_indicator.zscore_max is not None:
                                _kpiKI['maxValue'] = key_indicator.zscore_max
                    else:
                        if _type == self._AVERAGE_VALUE:
                            if key_indicator.average_value_min is not None:
                                _kpiKI['minValue'] = key_indicator.average_value_min
                            if key_indicator.average_value_max is not None:
                                _kpiKI['maxValue'] = key_indicator.average_value_max
                        elif _type == self._STANDARD_DEVIATION:
                            if key_indicator.standard_deviation_min is not None:
                                _kpiKI['minValue'] = key_indicator.standard_deviation_min
                            if key_indicator.standard_deviation_max is not None:
                                _kpiKI['maxValue'] = key_indicator.standard_deviation_max
                        elif _type == self._SHIFT:
                            if key_indicator.shift_min is not None:
                                _kpiKI['minValue'] = key_indicator.shift_min
                            if key_indicator.shift_max is not None:
                                _kpiKI['maxValue'] = key_indicator.shift_max
                    if 'kpis' not in data['task']['params']:
                        data['task']['params']['kpis'] = []
                    data['task']['params']['kpis'].append(_kpiKI)

        if (locally_increase_complexity):
            data['task']['params']['maxComplexity'] = max_complexity
            data['task']['params']['nbMinimizations'] = nb_minimizations
            data['task']['params']['coverageIncrement'] = coverage_increment
            msg += "\n\t- Max complexity: {} \n\t- Number of Minimizations: {} \n\t- Minimization \
            Coverage Increment: {}".format(max_complexity, nb_minimizations, coverage_increment)
        if (validate_stability):
            data['task']['params']['percentageSplit'] = split_ratio
            data['task']['params']['nbModels'] = nb_iterations
            data['task']['params']['purityTolerance'] = purity_tolerance
            msg += "\n\t- Percentage split: {} \n\t- Number of Iterations: {} \n\t- Purity Tolerance: {}".format(split_ratio, nb_iterations, purity_tolerance)

        print(msg)
        _ruleset = self.__api.Task.createtask(project_ID=self.__project_id, json=data)
        self.__api.handle_work_states(self.__project_id, work_type='learning', work_id=_ruleset.get('_id'))
        return self.get(name)

    @Helper.try_catch
    def filter(self):
        """
        Get all the rulesets of the project

        Returns:
            List of ruleset
        """
        from hypercube_api.hyper_api.dataset import DatasetFactory
        factory = DatasetFactory(self.__api, self.__project_id)
        ruleset_project = self.__api.Rules.getlearnings(project_ID=self.__project_id)
        return [Ruleset(self, self.__api, factory.get_by_id(ruleset.get('datasetId')), ruleset) for ruleset in ruleset_project]

    @Helper.try_catch
    def minimize(self, ruleset, minimization_name, score_to_minimize='Purity', increment_threshold=0.01):
        """
        Perform a minimzation on a give ruleset

        Args:
            ruleset (Ruleset): Ruleset to minimize
            minimization_name (str): Name of the new ruleset
            score_to_minimize (str): Score to apply the minimization, default is purity
            increment_threshold (float): Percentage increment of target samples that a new rule must bring to be added to the minimized ruleset.

        Return:
            Ruleset
        """
        json = {
            "type": "minimization",
            "datasetId": ruleset.dataset_id,
            "projectId": ruleset.project_id,
            "params": {
                "query": "tagsfilter={}".format(urllib.parse.quote(ruleset.name)),
                "taglist": [ruleset.name],
                "kpisList": ruleset.kpis,
                "kpiId": decode_kpiname_to_id(ruleset.kpis, score_to_minimize),
                "incrementThreshold": increment_threshold,
                "tag": minimization_name
            }
        }
        _ruleset = self.__api.Task.createtask(project_ID=ruleset.project_id, json=json)
        self.__api.handle_work_states(ruleset.project_id, work_type='minimization', work_id=_ruleset.get('_id'))
        return self.get(minimization_name)

    def get(self, name):
        """
        Get a ruleset by name

        Args:
            name (str): Name of the ruleset

        Returns:
            Ruleset
        """
        try:
            return [ruleset for ruleset in self.filter() if ruleset.name == name][0]
        except IndexError:
            return []

    @Helper.try_catch
    def get_by_id(self, id):
        """
        Get the ruleset matching the given ID or None if there is no match

        Returns:
            The Ruleset or None
        """
        rulesets = [ruleset for ruleset in self.filter() if ruleset.id == id]
        if rulesets:
            return rulesets[0]
        return None

    def get_or_create(self, dataset, name, target=None, purity_min=None, coverage_min=None, lift_min=None, zscore_min=None, average_value_min=None,
                      standard_deviation_max=None, shift_min=None, rule_complexity=2, quantiles=10,
                      enable_custom_discretizations=True, min_marginal_contribution=None, compute_other_key_indicators=None,
                      locally_increase_complexity=False, max_complexity=3, nb_minimizations=1, coverage_increment=0.01,
                      validate_stability=False, split_ratio=0.7, nb_iterations=1, purity_tolerance=0.1):
        """
        Get or create a ruleset, if the ruleset exists, only the name is mandatory

        Args:
            name (str): Name of the ruleset
            target (Target): Target to generate the ruleset
            purity_min (float): Minimum purity of rules, default is the entire dataset purity (discrete target only)
            coverage_min (int): Minimum coverage of the target population for each rule, default is 10 (discrete target only)
            lift_min (float): Minimum lift, default is 1 (discrete target only)
            zscore_min (float): Minimum Z-score, default is None (discrete target only)
            average_value_min (float): Minimum average value, default is average value of the target on the whole dataset (continuous target only)
            standard_deviation_max (float) : Maximum standard deviation, default is None (continuous target only)
            shift_min (float): Minimum shift, default is None (continuous target only)
            rule_complexity (int): Number of features considered for generating the association rules
            quantiles (int): Number of intervals the continuous variables are quantized in
            enable_custom_discretizations (boolean): use custom discretizations, eventually use "quantiles" parameter for remaining variables, default is True
            min_marginal_contribution (float): a new rule R', created by adding a new constraint to an existing rule R (and thus increasing its complexity),
                is added to the ruleset if and only if it increases the original purity of R by the minimum marginal contribution or more. Default is 0.1
            compute_other_key_indicators (list of KeyIndicatorOption): Compute other Key Indicators.
            locally_increase_complexity (bool): Enable the locally increase complexity when set as true.
            max_complexity (int): Maximum numbers of features per rule.
            nb_minimizations (int):Interate the minimization process.
            coverage_increment (float): Percetage increment of target samples that a new rule must bring to be added to the minimization ruleset.
            validate_stability (bool): Enable to split your dataset, add iteration and set a purity tolerance when set as true.
            split_ratio (float): The percentage for the split (Between 0 and 1).
            nb_iterations (int): Number of iterations wanted.
            purity_tolerance (float): Purity tolerence allowed (Between 0 and 1).

        Returns:
            Ruleset
        """

        for ruleset in dataset.rulesets:
            if (ruleset.name == name) and (ruleset.dataset_id == dataset.dataset_id):
                return ruleset

        return self.create(dataset, name, target, purity_min, coverage_min, lift_min, zscore_min, average_value_min, standard_deviation_max, shift_min,
                           rule_complexity, quantiles, enable_custom_discretizations,
                           min_marginal_contribution, compute_other_key_indicators, locally_increase_complexity, max_complexity,
                           nb_minimizations, coverage_increment, validate_stability, split_ratio, nb_iterations, purity_tolerance)


class Ruleset(Base):
    def __init__(self, factory, api, dataset, json_return):
        self.__api = api
        self.__factory = factory
        self.__json_returned = json_return
        self.__dataset = dataset
        self._is_deleted = False

    def __repr__(self):
        return """\n{} : {} <{}>\n""".format(
            self.__class__.__name__,
            self.name,
            self.id
        ) + ("\t<! This ruleset has been deleted>\n" if self._is_deleted else "") + \
            """\t- Dataset : {}\n\t- Rules count : {}\n\t- Created on : {}\n""".format(
            self.dataset_name,
            self.rules_count,
            self.created.strftime('%Y-%m-%d %H:%M:%S UTC'))

    # Property part
    @property
    def _json(self):
        return self.__json_returned

    @property
    def dataset_name(self):
        return self.__json_returned.get('datasetName')

    @property
    def name(self):
        return self.__json_returned.get('tag', {}).get('tagName')

    @property
    def kpis(self):
        return self.__json_returned.get('tag').get('kpis')

    @property
    def rules_count(self):
        return self.__json_returned.get('rulesCount')

    @property
    def dataset_id(self):
        return self.__json_returned.get('datasetId')

    @property
    def project_id(self):
        return self.__json_returned.get('projectId')

    @property
    def created(self):
        return self.str2date(self.__json_returned.get('lastChangeAt'), '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def id(self):
        return self.__json_returned.get('_id')

    # Method part
    @Helper.try_catch
    def get_params(self):
        if not self._is_deleted:
            return NotImplemented

    @Helper.try_catch
    def export(self):
        if not self._is_deleted:
            return NotImplemented

    @Helper.try_catch
    def delete(self):
        """
        Delete the ruleset
        """
        if not self._is_deleted:
            json = {
                '_id': self.id,
                'status': 'done',
                'tagName': self.name
            }
            self.__api.Rules.removealearning(project_ID=self.project_id, dataset_ID=self.dataset_id, json=json)
            self._is_deleted = True
        return self

    @Helper.try_catch
    def minimize(self, minimization_name, score_to_minimize='Purity', increment_threshold=0.01):
        """
        Function to apply a minimization on a ruleset

        Args:
            minimization_name (str): Name of the new ruleset
            score_to_minimize (str): Score to apply the minimization, default is purity
            increment_threshold (float): Percentage increment of target samples that a new rule must bring to be added to the minimized ruleset.

        Returns:
            Ruleset
        """
        if not self._is_deleted:
            return self.__factory.minimize(self, minimization_name, score_to_minimize, increment_threshold)

    @Helper.try_catch
    def get_rules(self, limit=100, sort=None, min_scores=None, max_scores=None, include_variables=None, exclude_variables=None):
        """
        Args:
            limit (int): Number of rules to get, default is 100
            sort (dict): Sorting the rules. Example {'score': 'size', 'asc': true}
            min_scores (dict): Set a minimum on a score. Example {'score': 'Purity', 'value': 0,563}
            max_scores (dict): Set a maximum on a score. Example {'score': 'Coverage', 'value': 300}
            include_variables (str): Variables to include
            exclude_variables (str): Variables to exclude

        Returns:
            List of rules
        """
        if not self._is_deleted:
            json = {'skip': 0,
                    'limit': limit,
                    'tagsfilter': self.name
                    }
            if sort:
                if 'score' in sort and 'asc' in sort:
                    var_name = decode_kpiname_to_id(self.kpis, sort['score'])
                    if sort['asc']:
                        json['sortasc'] = var_name
                    else:
                        json['sortdesc'] = var_name
                else:
                    raise ValueError('Wrong sort parameter, please follow this syntax : {\'score\': String, \'asc\': Boolean}')

            if min_scores:
                if isinstance(min_scores, dict):
                    min_scores = [min_scores]
                for score in min_scores:
                    if 'score' in score and 'value' in score:
                        var_name = decode_kpiname_to_id(self.kpis, score['score'])
                        json['min ' + var_name] = score['value']
                    else:
                        raise ValueError('Wrong min_score parameter, please follow this syntax : {\'score\': String, \'value\': Number}')

            if max_scores:
                if isinstance(max_scores, dict):
                    max_scores = [max_scores]
                for score in max_scores:
                    if 'score' in score and 'value' in score:
                        var_name = decode_kpiname_to_id(self.kpis, score['score'])
                        json['max ' + var_name] = score['value']
                    else:
                        raise ValueError('Wrong max_score parameter, please follow this syntax : {\'score\': String, \'value\': Number}')

            if include_variables:
                if isinstance(include_variables, str):
                    include_variables = [include_variables]
                json['varinclus'] = urllib.parse.quote(','.join(include_variables), safe='~()*!.\'')

            if exclude_variables:
                if isinstance(exclude_variables, str):
                    exclude_variables = [exclude_variables]
                json['varexclus'] = urllib.parse.quote(','.join(exclude_variables), safe='~()*!.\'')

            json_returned = self.__api.Rules.getrules(project_ID=self.project_id, dataset_ID=self.dataset_id, params=json).get('rules')
            return Rules(self.__api, json_returned, self.kpis, self.project_id, self.dataset_id)

    def predict(self, dataset, name, target, nb_minimizations=1, coverage_increment=0.01):
        """
        Create a prediction model from the ruleset

        Args:
            dataset (Dataset): Dataset to apply the prediction
            name (str): Name of the new model
            target (Target): Target used to generate the model
            nb_minimizations (int): Number of minimizations to perform on the ruleset, default is 1
            coverage_increment (float): Percentage increment of target samples that a new rule must bring to be added to the minimized ruleset,
                default is 0.01

        Returns:
            Model
        """
        if not self._is_deleted:
            return ModelFactory(self.__api, self.project_id).predict_from_ruleset(self.__dataset, dataset, self.name, name, target,
                                                                                  nb_minimizations, coverage_increment)
