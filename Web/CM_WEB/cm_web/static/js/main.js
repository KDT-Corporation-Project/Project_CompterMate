let term=500;

var $j = jQuery.noConflict(); // $j로 jQuery 객체 저장 -> 이거 없으면  충돌남
console.log("jQuery type:", typeof $j);  // jQuery가 올바르게 로드되었는지 확인

// $(document).ready 블록 내부에 있는 코드는 문서가 완전히 로드된 후 실행
$j(document).ready(function() {
  var graph1DataElement = $j('#graph1-data');
  var donut2DataElement = $j('#donut2-data');
  var donut3DataElement = $j('#donut3-data');

  // 그래프 초기화 
  if (graph1DataElement.length) {
    try {
      var graph1 = JSON.parse(graph1DataElement.text());
      Plotly.newPlot("chart1", graph1, {});  // 차트 생성
      console.log("chart ready");
    } catch (e) {
      console.error("Error parsing JSON data:", e);
    }
  } else {
    console.error("#graph1-data element not found");
  }

  // if (donut2DataElement.length) {
  //   try {
  //     var donut2 = JSON.parse(donut2DataElement.text());
  //     Plotly.newPlot("donut-chart-2", donut2, {});  // 차트 생성
  //     console.log("도넛2 ready");
  //   } catch (e) {
  //     console.error("Error parsing JSON data:", e);
  //   }
  // } else {
  //   console.error("도넛2 not found");
  // }

// 그래프 업데이트 
function updateChart(now_button) {
  // update_chart 뭔지 잘 보기 여러가지 version이 있어욤
  $j.getJSON('/update_chart', function(data) {
    // console.log("Data received:", data);  // 콘솔 로그 추가
    // console.log(data.msg);
    //
    try {
      var new_graph = JSON.parse(data.new_graphJSON);  // JSON 문자열을 객체로 변환
      Plotly.react("chart1", new_graph.data, new_graph.layout);  // 차트 업데이트
    } catch (e) {
      console.error("Error parsing JSON data:", e);
    }
  }).fail(function(jqXHR, textStatus, errorThrown) {
    console.error("AJAX request failed:", textStatus, errorThrown);  // 에러 로그 추가
  });
  }

// 도넛2 업데이트 
function updateDonut() {
  // update_donut
  $j.getJSON('/update_donut', function(data) {
    console.log("Data received:", data);  // 콘솔 로그 추가
    console.log(data.msg);

    try {
      var new_donut2 = JSON.parse(data.new_donut2JSON);  // JSON 문자열을 객체로 변환
      var new_donut3 = JSON.parse(data.new_donut3JSON);  // JSON 문자열을 객체로 변환
      Plotly.react("donut-chart-2", new_donut2.data, new_donut2.layout);  // 차트 업데이트
      Plotly.react("donut-chart-3", new_donut3.data, new_donut3.layout);  // 차트 업데이트
    } catch (e) {
      console.error("Error parsing JSON data:", e);
    }
  }).fail(function(jqXHR, textStatus, errorThrown) {
    console.error("AJAX request failed:", textStatus, errorThrown);  // 에러 로그 추가
  });
  }

  setInterval(updateChart, term); // 그래프 업데이트
  setInterval(updateDonut, term); // 도넛 업데이트
});
// $(document).ready 끝


//////////////////////////////////////////////////////////////////////////////////////////////////////////// 아래는 버튼에 대한 내용 
document.addEventListener("DOMContentLoaded", function() {
  // 버튼들을 가져옵니다.
  let buttons = document.querySelectorAll('.button_');

  // 각 버튼에 대해 클릭 이벤트 리스너를 추가합니다.
  buttons.forEach(button => {
    button.addEventListener("click", function() {
      let buttonValue = this.value; // 클릭된 버튼의 값 가져오기

      // 서버로 AJAX 요청을 보냅니다.
      let xhr = new XMLHttpRequest();
      xhr.open("POST", "/click_button"); // 요청을 보낼 URL을 지정합니다.
      xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
      xhr.onload = function() {
        if (xhr.status == 200) {
          console.log("Button value sent to server:", buttonValue);
        } else {
          console.error("Request failed:", xhr.status, xhr.statusText);
        }
      };
      xhr.onerror = function() {
        console.error("Network error");
      };
      xhr.send(JSON.stringify({ button: buttonValue })); // 버튼의 값을 JSON 형태로 서버에 전송합니다.
    });
  });
});

// 버튼 이벤트 active 삭제 
document.addEventListener("DOMContentLoaded", function() {
  let buttons = document.querySelectorAll('.button_');
  
  buttons.forEach(button => {
      button.addEventListener('click', function() {
          buttons.forEach(btn => btn.classList.remove('active'));
          this.classList.add('active');
      });
  });
});





// 게이지 JS (새로 추가됨)
document.addEventListener("DOMContentLoaded", function() {
  function createGaugeChart(id, title, range, steps) {
    var data = [{
      type: "indicator",
      mode: "gauge+number+delta",
      value: 0,
      title: { text: title, font: { size: 15 }},
      delta: { reference: (range[0]+range[1])*0.5 },
      gauge: {
        axis: { range: range, startangle: -135, endangle: 135 }, // 각도를 270도로 설정
        bar: { color: "white" },
        steps: steps,
        bordercolor: 'rgba(0,0,0,0)', // 테두리선을 투명하게 설정



      }
    }];

    var layout = {
      width: 280,
      height: 280,
      paper_bgcolor: 'rgba(0,0,0,0)', // 배경을 투명하게 설정
      plot_bgcolor: 'rgba(0,0,0,0)', // 배경을 투명하게 설정
      margin: { t: 100, b: 100, l:0, r:0},
    };

    Plotly.newPlot(id, data, layout);
  }

  function updateGaugeChart(id, value) {
    var dataUpdate = {
      value: [value]

    };
    console.log(dataUpdate)
    Plotly.update(id, dataUpdate);
  }

  function createAllGauges() {
    createGaugeChart('gauge-chart-1', '중량 무게(g)', [2, 4], [
      { range: [2, 2.4], color: "#ef3d21" },
      { range: [2.4, 2.8], color: "#FFA500" },
      { range: [2.8, 3.2], color: "#1FA680" },
      { range: [3.2, 3.6], color: "#FFA500" },
      { range: [3.6, 4], color: "#ef3d21" }
    ]);
    createGaugeChart('gauge-chart-2', '칼날 RPM', [130, 230], [
      { range: [130, 150], color: "#ef3d21" },
      { range: [150, 170], color: "#FFA500" },
      { range: [170, 190], color: "#1FA680" },
      { range: [190, 210], color: "#FFA500" },
      { range: [210, 230], color: "#ef3d21" }
    ]);
    createGaugeChart('gauge-chart-3', '노즐 온도(℃)', [45, 95], [
      { range: [45, 55], color: "#ef3d21" },
      { range: [55, 65], color: "#FFA500" },
      { range: [65, 75], color: "#1FA680" },
      { range: [75, 85], color: "#FFA500" },
      { range: [85, 95], color: "#ef3d21" }
    ]);
    createGaugeChart('gauge-chart-4', '스크류 온도(℃)', [45, 95], [
      { range: [45, 55], color: "#ef3d21" },
      { range: [55, 65], color: "#FFA500" },
      { range: [65, 75], color: "#1FA680" },
      { range: [75, 85], color: "#FFA500" },
      { range: [85, 95], color: "#ef3d21" }
    ]);
    createGaugeChart('gauge-chart-5', '챔버 온도(℃)', [45, 95], [
      { range: [45, 55], color: "#ef3d21" },
      { range: [55, 65], color: "#FFA500" },
      { range: [65, 75], color: "#1FA680" },
      { range: [75, 85], color: "#FFA500" },
      { range: [85, 95], color: "#ef3d21" }
    ]);
    createGaugeChart('gauge-chart-6', '스크류 속도(E)', [5.5, 10.5], [
      { range: [5.5, 6.5], color: "#ef3d21" },
      { range: [6.5, 7.5], color: "#FFA500" },
      { range: [7.5, 8.5], color: "#1FA680" },
      { range: [8.5, 9.5], color: "#FFA500" },
      { range: [9.5, 10.5], color: "#ef3d21" }
    ]);
  }

  function updateAllGauges() {
    $j.getJSON('/update_gauges', function(data) {
      updateGaugeChart('gauge-chart-1', data['중량 예측']);
      updateGaugeChart('gauge-chart-2', data['칼날 속도']);
      updateGaugeChart('gauge-chart-3', data['노즐 온도']);
      updateGaugeChart('gauge-chart-4', data['스크류 온도']);
      updateGaugeChart('gauge-chart-5', data['챔버 온도']);
      updateGaugeChart('gauge-chart-6', data['스크류 속도']);
      // console.log(data.c_temp_pv, data.k_rpm_pv, data.n_temp_pv, data.s_temp_pv, data.scale_pv, data.E_scr_pv)
    }).fail(function(jqXHR, textStatus, errorThrown) {
      console.error("AJAX request failed:", textStatus, errorThrown);
    });
  }

  createAllGauges();
  setInterval(updateAllGauges, term);
});

// let term=500;