# data_processing 클래스로 관리하는 것이 아닌, method 로 호출하도록 

import numpy as np
from utils.decorators import data_validation  # 데코레이터 함수 임포트

# 데이터 검증 데코레이터
@data_validation
def missing_value_del(data, p_axis=1):
    """
    결측치가 제거된 데이터 반환

    Args:
        data(ndarray) : 결측치가 존재하는 데이터
        p_axis(default = 1) : 1 - 결측치 행의 제거, 0 - 결측치 열의 제거

    Returns:
        결측치가 제거된 ndarray

    """

    # ~ : 결과를 부정, 
    # np.isnan(data) : NaN 값의 True
    # .any : 각 행, 열에 대해 결측치가 하나라도 있으면 True 로 나타낸다.
    # data[] : True 값만 남긴다.
    return data[~np.isnan(data).any(axis=p_axis)]

@data_validation
def missing_value_cha(data, is_zero = False, is_ave = False, p_axis = 0):
    """
    결측치의 대체

    Args:
        data(ndarray) : 결측치가 존재하는 데이터
        is_zero(default = False) : 결측치를 0으로 대체
        is_ave(default = False) : 결측치를 평균으로 대체
        axis(default = 0) : 평균 대체 시 기준이 되는 열, 행의 선택

    Returns:
        결측치가 제거된 ndarray
    """

    if(is_zero):
        data = np.nan_to_num(data, nan = 0)
    
    elif(is_ave):
        mean = np.nanmean(data, axis = p_axis)
        inds = np.where(np.isnan(data))
        data[inds] = np.take(mean, inds[0])

    return data

@data_validation
def duplicate_data_del(data):
    """
    중복 데이터의 제거

    Args:
        data(ndarray) : 중복이 존재하는 데이터

    Returns:
        중복이 제거된 ndarray
    """

    return np.unique(data, axis=0)

@data_validation
def min_max_normalization(data):
    """
    min, max 정규화, 데이터를 0과 1 사이의 값으로 변환, 
    데이터가 특정 범위 내에 존재하도록 보장
    값의 범위를 제한해야 하거나, 양수 값만 다루는 경우

    Args:
        data(ndarray) : 정규화할 데이터
    
    Returns:
        정규화된 ndarray
    """
    data_min = np.min(data, axis=0)  # 각 열의 최소값
    data_max = np.max(data, axis=0)  # 각 열의 최대값
    return (data - data_min) / (data_max - data_min)

@data_validation
def standardize(data):
    """
    표준화, 평균이 0이고 표준편차가 1이 되도록 변환
    각 데이터 포인트가 평균에서 몇 표준편차만큼 떨어져 있는지 계산 가능
    X' = X - mu / sigma

    Args:
        data(ndarray) : 데이터
    
    Returns:
        표준화된 ndarray
    """
    data_mean = np.mean(data, axis=0)  # 각 열의 평균
    data_std = np.std(data, axis=0)    # 각 열의 표준편차
    return (data - data_mean) / data_std

@data_validation
def max_abs_normalize(data):
    """
    각 특성의 최대 절대값이 1이 되도록 데이터 스케일링 [-1, 1] 범위
    데이터에 음수 값이 포함된 경우 유용, 양.음수 관계의 유지

    Args:
        data(ndarray) : 데이터
    
    Returns:
        ndarray
    """

    data_max_abs = np.max(np.abs(data), axis=0)
    return data / data_max_abs

@data_validation
def robust_normalize(data):
    """
    중앙값과 InterquartileRange, 1분위수와 3분위수 간의 범위)를 사용하여 데이터 정규화
    데이터에 이상치가 있을 때, 이상치의 영향을 줄이기 위해 사용
    데이터에 이상치가 많을 경우 유용

    Args:
        data(ndarray) : 데이터
    
    Returns:
        ndarray
    """
    data_median = np.median(data, axis=0)
    data_iqr = np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0)
    return (data - data_median) / data_iqr

@data_validation
def unit_vector_normalize(data):
    """
    각 데이터 벡터를 그 벡터의 L2 norm 으로 나누어 벡터의 크기가 1이 되도록 만든다.
    데이터를 방향성만 유지하고 크기는 표준화할 때 유용

    Args:
        data(ndarray) : 데이터
    
    Returns:
        ndarray
    """
    norm = np.linalg.norm(data, axis=0)
    return data / norm

@data_validation
def log_normalize(data):
    """
    데이터의  분포가 크게 치우쳐 있는 경우, 로그 변환을 통해 데이터 분포 조정
    양수 데이터에만 적용 가능
    데이터의 분포가 비대칭적이거나 양수 데이터의 범위가 넓을 때 유용

    Args:
        data(ndarray) : 데이터
    
    Returns:
        ndarray
    """
    return np.log1p(data)

@data_validation
def one_hot_encoding(data):
    """
    각 카테코리 값을 이진 벡터로 변환하는 방법
    Args:
        data(ndarray) : 데이터
    
    Returns:
        ndarray (p,n)
    """
    # 고유한 카테고리 찾기
    unique_categories = np.unique(data)

    # 원-핫 인코딩
    one_hot_encoded = np.zeros((data.shape[0], unique_categories.shape[0]))

    # 각 카테고리 위치에 1 설정
    for i, category in enumerate(data):
        one_hot_encoded[i, np.where(unique_categories == category)[0][0]] = 1

    return one_hot_encoded

@data_validation
def label_encoding(data):
    """
    각 카테고리 값을 정수로 변환
    Args:
        data(ndarray) : 데이터
    
    Returns:
        ndarray (p,n)
    """
    # 고유한 카테고리 찾기
    unique_categories = np.unique(data)

    # 레이블 인코딩
    label_encoded = np.array([np.where(unique_categories == category)[0][0] for category in data])

    return label_encoded

@data_validation
def shuffle_split_data(train_data, target_data, split_size):
    """
    데이터셋의 분할
    Args:
        train_data(ndarray) : 훈련 데이터
        target_data(ndarray) : 타겟 데이터
        split_size : 분할 크기
    
    Returns:
        ndarray1 : 분할 데이터 1
        ndarray2 : 분할 데이터 2
    """
    indices = np.arrange(train_data.shape[0])
    np.random.shuffle(indices)

    # 분할 인덱스 계산
    train_size = int(train_data.shape[0] * split_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # 훈련 세트와 테스트 세트로 데이터 분할
    X_train, X_test = train_data[train_indices], train_data[test_indices]
    y_train, y_test = target_data[train_indices], target_data[test_indices]

    return X_train, X_test, y_train, y_test

