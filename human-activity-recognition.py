import pandas as pd
import matplotlib.pyplot as plt

# UCI HAR Dataset/features.txt's data is separated with spaces. Loading this in DataFrame type
feature_name_df = pd.read_csv("./UCI HAR Dataset/features.txt", sep="\s+", header=None, names=["column_index", "column_name"])
feature_name = feature_name_df.iloc[:, 1].values.tolist()
print("Extracting 10 rows from total features data :", feature_name[:10])
feature_dup_df = feature_name_df.groupby("column_name").count()
print(feature_dup_df)
print(feature_dup_df[feature_dup_df["column_index"] > 1].count())
print(feature_dup_df[feature_dup_df["column_index"] > 1].head(10))

df = pd.DataFrame([["a"], ["a"], ["a"], ["b"], ["b"], ["a"]], columns=["A"])
print(df)
df.groupby("A").cumcount()

# 원본 데이터에 중복된 Feature 명으로 인하여 신규 버전의 Pandas에서 Duplicate name 에러를 발생.
# 중복 Feature 명에 대해서 원본 feature 명에 "_1(또는 2)"를 추가로 부여하는 함수인 get_new_feature_name_df() 생성
def get_new_feature_name_df(old_feature_name_df):
    feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby("column_name").cumcount(), columns=["dup_cnt"])
    feature_dup_df = feature_dup_df.reset_index()
    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(), feature_dup_df, how="outer")
    new_feature_name_df['column_name'] = new_feature_name_df[["column_name", "dup_cnt"]].apply(lambda x : x[0] + "_" + str(x[1]) if x[1] > 0 else x[0], axis=1)

    new_feature_name_df = new_feature_name_df.drop(["index"], axis=1)
    return new_feature_name_df

def get_human_dataset():
    feature_name_df = pd.read_csv("./UCI HAR Dataset/features.txt", sep="\s+", header=None, names=["column_index", "column_name"])

    # 중복된 피쳐명을 수정하는 get_new_feature_name_df()를 이용, 신규 피쳐명 DataFrame 생성
    new_feature_name_df = get_new_feature_name_df(feature_name_df)

    feature_name = new_feature_name_df.iloc[:, 1].values.tolist()

    X_train = pd.read_csv("./UCI HAR Dataset/train/X_train.txt", sep="\s+", names=feature_name)
    X_test = pd.read_csv("./UCI HAR Dataset/test/X_test.txt", sep="\s+", names=feature_name)

    y_train = pd.read_csv("./UCI HAR Dataset/train/y_train.txt", sep="\s+", header=None, names=["action"])
    y_test = pd.read_csv("./UCI HAR Dataset/test/y_test.txt", sep="\s+", header=None, names=["action"])

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_human_dataset()

print("## 학습 피쳐 데이터셋 info()")
print(X_train.info())

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

dt_clf = DecisionTreeClassifier(random_state=156)
dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print("결정 트리 예측 정확도: {0:.4f}".format(accuracy))
print("DecisionTreeClassifier 기본 하이퍼 파라미터 : \n", dt_clf.get_params())

params = {
    "max_depth" : [6, 8, 10, 12, 16, 20, 24],
    "min_samples_split" : [16]
}
grid_cv = GridSearchCV(dt_clf, param_grid=params, scoring="accuracy", cv=5, verbose=1)
grid_cv.fit(X_train, y_train)
print("GridSearchCV 최고 평균 정확도 수치 : {0:.4f}".format(grid_cv.best_score_))
print("GridSearchCV 최적 하이퍼 파라미터 :", grid_cv.best_params_)

# GridSearchCV 객체의 cv_results_ 속성을 DataFrame으로 생성.
cv_results_df = pd.DataFrame(grid_cv.cv_results_)
# max_depth 파라미터 값과 그때의 테스트(Evaluation)세트, 학습 데이터 세트의 정확도 수치 추출
print(cv_results_df[["param_max_depth", "mean_test_score"]])

for depth in params["max_depth"]:
    dt_clf = DecisionTreeClassifier(max_depth=depth, min_samples_split=16, random_state=156)
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print("max_depth = {0} 정확도: {1:.4f}".format(depth, accuracy))

params = {
    "max_depth" : [8, 12, 16, 20],
    "min_samples_split": [16, 24]
}
grid_cv = GridSearchCV(dt_clf, param_grid=params, scoring="accuracy", cv=5, verbose=1)
grid_cv.fit(X_train, y_train)
print("GridSearchCV 최고 평균 정확도 수치 : {0:.4f}".format(grid_cv.best_score_))
print("GridSearchCV 최적 하이퍼 파라미터 :", grid_cv.best_params_)

best_df_clf = grid_cv.best_estimator_
pred1 = best_df_clf.predict(X_test)
accuracy = accuracy_score(y_test, pred1)
print("결정 트리 예측 정확도 : {0:.4f}".format(accuracy))

ftr_importances_values = best_df_clf.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index=X_train.columns)
ftr_importances.sort_values(ascending=False)

import seaborn as sns

ftr_importances_values = best_df_clf.feature_importances_
# Top 중요도로 정렬을 쉽게 하고, Seaborn의 막대그래프로 쉽게 표현하기 위해 Series 변환
ftr_importances = pd.Series(ftr_importances_values, index=X_train.columns)
# 중요도값 순으로 Series 정렬
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
plt.figure(figsize=(8,6))
plt.title("Feature importances Top 20")
sns.barplot(x=ftr_top20, y=ftr_top20.index)
plt.show()
