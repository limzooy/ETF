# ETF 투자 분석 대시보드

Python FastAPI와 Yahoo Finance API를 이용하여 주요 ETF의 기술적 지표를 분석하고, 매매 시점을 참고할 수 있는 웹 대시보드입니다.

## 주요 기능

-   실시간 ETF 가격 및 시장 데이터 조회
-   이동평균선, 볼린저밴드 등 보조지표 시각화
-   매수/매도 전략 신호 및 기술적 분석 리뷰 제공

## 실행 방법

1.  저장소를 복제(clone)합니다.
    ```bash
    git clone [https://github.com/limzooy/ETF.git](https://github.com/limzooy/ETF.git)
    cd YourRepoName
    ```
2.  가상환경을 생성하고 활성화합니다.
    ```bash
    python -m venv venv
    .\venv\Scripts\Activate
    ```
3.  필요한 라이브러리를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```
4.  FastAPI 서버를 실행합니다.
    ```bash
    uvicorn main:app --reload
    ```
5.  웹 브라우저에서 `http://127.0.0.1:8000` 주소로 접속합니다.