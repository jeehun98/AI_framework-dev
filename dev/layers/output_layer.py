class LayerOutput:
    def __init__(self, shape, data=None, metadata=None):
        """
        레이어의 출력 객체.
        
        Parameters:
        - shape: 출력의 형태
        - data: 출력 데이터 (기본값: None)
        - metadata: 출력과 관련된 추가 정보 (기본값: None)
        """
        self.shape = shape
        self.data = data  # 실제 데이터, 아직 할당되지 않은 경우 None
        self.metadata = metadata if metadata is not None else {}  # 추가 메타데이터

    def get_shape(self):
        """출력의 형태 반환"""
        return self.shape

    def set_data(self, data):
        """출력 데이터를 설정"""
        self.data = data

    def get_data(self):
        """출력 데이터를 반환"""
        return self.data

    def add_metadata(self, key, value):
        """추가적인 메타데이터를 설정"""
        self.metadata[key] = value

    def get_metadata(self, key, default=None):
        """특정 메타데이터를 반환. 없으면 기본값 반환"""
        return self.metadata.get(key, default)

    def __repr__(self):
        return f"LayerOutput(shape={self.shape}, data={self.data}, metadata={self.metadata})"
