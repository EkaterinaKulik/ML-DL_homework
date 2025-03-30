import numpy as np
from collections import Counter
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    sort_idx = np.argsort(feature_vector)
    feature_sorted = feature_vector[sort_idx]
    target_sorted = target_vector[sort_idx]

    n = len(feature_sorted)

    diff = feature_sorted[1:] > feature_sorted[:-1]
    if not np.any(diff):
        return np.array([]), np.array([]), None, None

    thresholds = (feature_sorted[:-1] + feature_sorted[1:]) / 2
    thresholds = thresholds[diff]

    cumsum_ones = np.cumsum(target_sorted == 1)
    total_ones = cumsum_ones[-1]
    total_zeros = n - total_ones

    split_indices = np.flatnonzero(diff)
    R_l_sizes = split_indices + 1
    R_r_sizes = n - R_l_sizes

    ones_left = cumsum_ones[split_indices]
    ones_right = total_ones - ones_left
    zeros_left = R_l_sizes - ones_left
    zeros_right = R_r_sizes - ones_right

    p1_left = ones_left / R_l_sizes
    p0_left = zeros_left / R_l_sizes
    p1_right = ones_right / R_r_sizes
    p0_right = zeros_right / R_r_sizes

    H_left = 1 - p1_left**2 - p0_left**2
    H_right = 1 - p1_right**2 - p0_right**2

    ginis = - (R_l_sizes / n * H_left + R_r_sizes / n * H_right)

    max_idx = np.argmax(ginis)
    gini_best = ginis[max_idx]
    threshold_best = thresholds[max_idx]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):

        if self._max_depth is not None and depth >= self._max_depth:  
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
      
        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split: 
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return 

        if np.all(sub_y == sub_y[0]): #проверка должна выполняться на соотвествие всех элементов, а не только первого элемента
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        n = len(sub_y) 
        for feature in range(1, sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            #словарь для категориальных признаков в этой части кода определять не нужно

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_count / (current_click if current_click != 0 else 0.000001) #если current_click == 0, то будет ошибка в вычислении
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list((map(lambda x: categories_map[x], sub_X[:, feature])))) #map возращает итератор, нужно обернуть в list для np.array
            else:
                raise ValueError

            if len(feature_vector) == 3:
                continue
            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if threshold is None or gini is None:
              continue

            left_mask = feature_vector < threshold
            left_count = np.sum(left_mask)
            right_count = n - left_count

            if self._min_samples_leaf is not None:
                if left_count < self._min_samples_leaf or right_count < self._min_samples_leaf:
                    continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical": #ранее было объявлено как categorical с маленькой буквы
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0] #нам нужен только класс
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth=depth+1)  
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth=depth+1) 
 #правое дерево должно использовать sub_y[np.logical_not(split)] для деления на подвыборку

    def _predict_node(self, x, node):

        if node["type"] == "terminal":
          return node["class"]
    
    
        feature = node["feature_split"]
        feature_type = self._feature_types[feature]
    
        if feature_type == "real":
          if x[feature] < node["threshold"]:
            return self._predict_node(x, node["left_child"])
          else:
            return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
          if x[feature] in node["categories_split"]:
            return self._predict_node(x, node["left_child"])
          else:
            return self._predict_node(x, node["right_child"])
        else:
          raise ValueError("Unknown feature type encountered during prediction.")

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, depth=0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

class LinearRegressionTree(DecisionTree):

    def __init__(self, feature_types, base_model_type=LinearRegression, max_depth=None, min_samples_split=None, min_samples_leaf=None, n_quantiles=10):
        super().__init__(feature_types, max_depth, min_samples_split, min_samples_leaf)
        self.base_model_type = base_model_type
        self.n_quantiles = n_quantiles  

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if self._max_depth is not None and depth >= self._max_depth:
            self._make_linear_leaf(sub_X, sub_y, node)
            return
        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            self._make_linear_leaf(sub_X, sub_y, node)
            return
        if np.all(sub_y == sub_y[0]):
            self._make_linear_leaf(sub_X, sub_y, node)
            return

        feature_best, threshold_best, loss_best, split_mask = self._find_best_split_linear(sub_X, sub_y)

        if feature_best is None:
            self._make_linear_leaf(sub_X, sub_y, node)
            return

        left_count = np.sum(split_mask)
        right_count = len(sub_y) - left_count
        if (self._min_samples_leaf is not None) and (left_count < self._min_samples_leaf or right_count < self._min_samples_leaf):
            self._make_linear_leaf(sub_X, sub_y, node)
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        node["threshold"] = threshold_best
        node["left_child"], node["right_child"] = {}, {}

        self._fit_node(sub_X[split_mask], sub_y[split_mask], node["left_child"], depth=depth+1)
        self._fit_node(sub_X[~split_mask], sub_y[~split_mask], node["right_child"], depth=depth+1)

    def _make_linear_leaf(self, sub_X, sub_y, node):
        node["type"] = "terminal"
        model = self.base_model_type()
        model.fit(sub_X, sub_y)
        node["model"] = model

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["model"].predict(x.reshape(1, -1))[0]

        feature = node["feature_split"]
        threshold = node["threshold"]
        if x[feature] < threshold:
            return self._predict_node(x, node["left_child"])
        else:
            return self._predict_node(x, node["right_child"])

    def _find_best_split_linear(self, X, y):
        n, m = X.shape
        best_feature = None
        best_threshold = None
        best_loss = np.inf
        best_split = None

        for feature in range(m):
            unique_values = np.unique(X[:, feature])
            if len(unique_values) == 1:
                continue

            quantiles = np.linspace(0, 1, self.n_quantiles+2)[1:-1] 
            thresholds = np.quantile(unique_values, quantiles)

            for thr in thresholds:
                left_mask = X[:, feature] < thr
                left_count = np.sum(left_mask)
                right_count = n - left_count
                if left_count == 0 or right_count == 0:
                    continue

                model_left = self.base_model_type()
                model_left.fit(X[left_mask], y[left_mask])
                y_pred_left = model_left.predict(X[left_mask])
                mse_left = mean_squared_error(y[left_mask], y_pred_left)

                model_right = self.base_model_type()
                model_right.fit(X[~left_mask], y[~left_mask])
                y_pred_right = model_right.predict(X[~left_mask])
                mse_right = mean_squared_error(y[~left_mask], y_pred_right)

                w_left = left_count / n
                w_right = right_count / n
                loss = w_left * mse_left + w_right * mse_right

                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature
                    best_threshold = thr
                    best_split = left_mask

        return best_feature, best_threshold, best_loss, best_split
