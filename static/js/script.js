// script.js
// 이 스크립트는 모든 동적인 대시보드 기능을 처리합니다.

// 전역 변수 설정
const etfTickers = ['069500', '292150'];
const etfNames = {
    '069500': 'KODEX 200',
    '292150': 'TIGER TOP10'
};
let currentTickerIndex = 0;
let currentSource = 'yahoo'; // 'yahoo' 또는 'krx'
let etfData = {}; // 데이터를 저장할 전역 객체

// 차트 인스턴스
let etfChart, simulatorChart;

// HTML 요소 선택자
// ... (이하 모든 JavaScript 코드)