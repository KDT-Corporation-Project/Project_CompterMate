from flask import Blueprint, redirect, jsonify
# Response, request => HTML 응답 요청을 처리하기 위함 
# render_template => HTML 파일을 렌더링
from flask import Flask, render_template, Response, request
import pandas as pd 
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


WINDOW_SIZE = 50
cnt = 0
wrong_cnt = 0

bp = Blueprint("main", __name__, url_prefix="/")

CM_DF = pd.read_csv("./cm_web/static/data/result_ver1.csv")
DATA_LENGTH = len(CM_DF)


graph_idx = 2 # 기본값 => rpm
name_list = ("E_scr", "c_temp", "k_rpm", "n_temp", "s_temp")

E_scr_pv = CM_DF["E_scr_pv"]
E_scr_sv = CM_DF["E_scr_sv"]

c_temp_pv = CM_DF["c_temp_pv"]
c_temp_sv = CM_DF["c_temp_sv"]

k_rpm_pv = CM_DF["k_rpm_pv"]
k_rpm_sv = CM_DF["k_rpm_sv"]

n_temp_pv = CM_DF["n_temp_pv"]
n_temp_sv = CM_DF["n_temp_sv"]

s_temp_pv = CM_DF["s_temp_pv"]
s_temp_sv = CM_DF["s_temp_sv"]

scale_pv = CM_DF["scale_pv"]

sr_list = ((E_scr_pv, E_scr_sv),
           (c_temp_pv, c_temp_sv),
           (k_rpm_pv, k_rpm_sv),
           (n_temp_pv, n_temp_sv),
           (s_temp_pv, s_temp_sv))
yaxis_list = ((0, 10),
              (55, 75),
              (120, 220),
              (55, 75),
              (55, 75))
x_values = np.linspace(-WINDOW_SIZE, 0, WINDOW_SIZE, endpoint=False)
x_labels = [f"{int(x)}초" if x != 0 else "현재" for x in x_values]

# 5초 단위로 x축 레이블 설정
tickvals = list(range(0, WINDOW_SIZE, 5)) + [WINDOW_SIZE-1]  # 0부터 WINDOW_SIZE까지 5 간격으로, 마지막에 0초전 추가
ticktext = [x_labels[i] for i in tickvals]
ticktext[0] = " "
'''
E_scr_pv,E_scr_sv, -> 0, 10
c_temp_pv,c_temp_sv -> 65 75
k_rpm_pv,k_rpm_sv -> 100 200
n_temp_pv,n_temp_sv -> 65 75
s_temp_pv,s_temp_s ->  65 75
'''

k_name_list = ["스크류 속도", "챔버 온도", "칼날 속도", "노즐 온도", "스크류 온도"]

# 각 column의 성분을 그려주는 그래프 
def draw_graph():
    global WINDOW_SIZE
    global CM_DF
    global cnt
    global x_labels

    col_pv = sr_list[graph_idx][0]
    col_sv = sr_list[graph_idx][1]
    new_fig = px.line(x=range(WINDOW_SIZE), y=col_pv[cnt:cnt+WINDOW_SIZE], markers=True)
    new_fig.update_traces(name="pv", showlegend=True, marker=dict(opacity=0.5)) # 기본 상태 그래프는 showlegend 안하면 밑에 sv 범례만 뜸 
    # 알파값 높을수록 진함

    # setting value 그래프 추가 
    new_fig.add_trace(go.Scatter(x=list(range(WINDOW_SIZE)), y=col_sv[cnt:cnt+WINDOW_SIZE], name="sv"))

    # 배경을 투명하게 설정, 주변부 없애기, 축범위 지정, 레이아웃 세로 길이 조금 줄임
    new_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          margin=dict(t=30, b=30, l=100, r=0),
                          xaxis_range=[0, WINDOW_SIZE],  
                          yaxis_range=yaxis_list[graph_idx],
                          height=300,
                          xaxis_title="경과시간 (초)",
                          yaxis_title=k_name_list[graph_idx] # y축 이름 설정
                          )
    
    # x축 레이블 설정
    new_fig.update_xaxes(
        tickvals = tickvals,
        ticktext= ticktext
    )
    
    cnt += 1
    new_graphJSON = json.dumps(new_fig, cls=plotly.utils.PlotlyJSONEncoder)
    return new_graphJSON
    
# dash 도넛 차트 그리는 함수 
def draw_donut(title, colors):
    global cnt
    global DATA_LENGTH
    global wrong_cnt
    
    if title == "불량률":
        if scale_pv[cnt] <= 2.9 or scale_pv[cnt] >= 3.09:
            wrong_cnt += 1
    # col_pv = sr_list[graph_idx][0]
    # col_sv = sr_list[graph_idx][1]
    
    # 불량품 그래프 
    new_fig = go.Figure(data=[go.Pie(
    values=[wrong_cnt, DATA_LENGTH - wrong_cnt] if title=="불량률" else [cnt, DATA_LENGTH - cnt],
    labels=['Used', 'Remaining'],
    marker=dict(colors=colors),
    textinfo='none',
    hole=.4,
    rotation=0,
    direction='clockwise',
    type='pie'
    )])
    

    # 배경을 투명하게 설정, 주변부 없애기, 축범위 지정, 레이아웃 세로 길이 조금 줄임
    new_fig.update_layout(
        title=dict(
            text=title,
            x=0.5,  # 중앙 정렬
            xanchor='center',  # 중앙 정렬
            font=dict(size=15)  # 제목 폰트 크기
        ),
        height=245,
        width=245,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',  # 배경을 투명하게 설정
        plot_bgcolor='rgba(0,0,0,0)',  # 배경을 투명하게 설정
        margin=dict(t=40, b=0, l=0, r=0),
        annotations=[
            dict(
                font=dict(size=15),
                showarrow=False,
                text=str(wrong_cnt) if title=="불량률" else str(cnt),
                x=0.5,
                y=0.5
            )
        ]
    )
    
    new_donutJSON = json.dumps(new_fig, cls=plotly.utils.PlotlyJSONEncoder)
    return new_donutJSON
    

# @=>데코레이터 
@bp.route("/")
def index():
    return render_template("job_choice.html")


# dashboard =========================================================================
@bp.route("/dash")
def dash():
    # global WINDOW_SIZE
    # global CM_DF
    # global cnt
    print("그래프 초기화")  # 로그 추가 -> python terminal에서 확인
    
   
    new_graphJSON = draw_graph()
    
    return render_template("dash.html", title="Home", graph1JSON=new_graphJSON)

# total + 누적 그래프 생성 
@bp.route("/total")
def total():
    global scale_pv
    # Calculate the accumulated values with 3 subtracted from each element
    accumulated_values = []
    accumulated_sum = 0
    for value in scale_pv:
        accumulated_sum += (value - 3)
        accumulated_values.append(accumulated_sum)

    # Create a scatter plot
    accumulate_fig = go.Figure()

    # Add the area fill
    accumulate_fig.add_trace(go.Scatter(
        x=list(range(1, len(accumulated_values) + 1)),
        y=accumulated_values,
        mode='lines', # +markers
        fill='tozeroy',  # Fill to the x-axis
        fillcolor='rgba(0, 0, 255, 0.3)',  # Blue color with transparency
        name='Accumulated Values'
    ))

    # Update layout for transparent background and tight layout
    accumulate_fig.update_layout(
        xaxis_title='기간',
        yaxis_title='누적값',
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent background
        width=1500, height=300,  # Reduced size
        margin=dict(l=0, r=0, t=20, b=20) # Tight layout
    )
    accumulate_graphJSON = json.dumps(accumulate_fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template("total.html", accumulate_graphJSON=accumulate_graphJSON)


@bp.route("/learning")
def learning():
    return render_template("learning.html")


# 차트는 여기서 업데이트됨
@bp.route('/update_chart')
def update_chart():
    # global graph_idx 
    # global sr_list # 쌍으로 담겨 있음 sr_list[graph_idx]로 쌍에 접근 가능 
    # global WINDOW_SIZE
    # global CM_DF
    # global cnt
    
    # print("Updating chart1...")  # 로그 추가 -> python terminal에서 확인
   
   
    new_graphJSON = draw_graph()
    # cnt += 1 # WINDOW 이동
    return jsonify({"new_graphJSON": new_graphJSON, "msg" : "화이팅!"})


# 규량님 거 JS to Python : 도넛 차트==================================================================================
@bp.route('/update_donut')
def update_donut():
    print("도넛 업데이트 중...")
    new_donut2JSON = draw_donut("생산량", ['#1FA680', '#ffffff'])
    new_donut3JSON = draw_donut("불량률", ['#FFA500', '#ffffff'])
    return jsonify({"new_donut2JSON": new_donut2JSON, "new_donut3JSON":new_donut3JSON," msg" : "도넛 차트 data.msg"})


# 규량 : 계기판 표현=================================================================================
@bp.route('/update_gauges')
def update_gauges():
    global cnt
    global WINDOW_SIZE
    data = {
        "챔버 온도": float(c_temp_pv[cnt+WINDOW_SIZE]),
        "칼날 속도": float(k_rpm_pv[cnt+WINDOW_SIZE]),
        "노즐 온도": float(n_temp_pv[cnt+WINDOW_SIZE]),
        "스크류 온도": float(s_temp_pv[cnt+WINDOW_SIZE]),
        "중량 예측": float(scale_pv[cnt+WINDOW_SIZE]),
        "스크류 속도": float(E_scr_pv[cnt+WINDOW_SIZE])
    }
    #cnt = (cnt + 1) % len(c_temp_pv)  # 데이터를 순환하도록 cnt를 리셋합니다.
    # print('게이지 오류!',c_temp_pv[cnt+WINDOW_SIZE])
    return jsonify(data)


################################################### 값 확인 
@bp.route('/click_button', methods=['POST'])
def click_button():
    global graph_idx
    global name_list
    data = request.get_json()
    print(data)
    name_list = ["E_scr", "c_temp", "k_rpm", "n_temp", "s_temp"]


    for i in range(len(name_list)):
        if data["button"] == name_list[i]:
            graph_idx = i
    '''
    E_scr_pv,E_scr_sv,
    c_temp_pv,c_temp_sv,
    k_rpm_pv,k_rpm_sv,       
    n_temp_pv,n_temp_sv,
    s_temp_pv,s_temp_sv
    '''
    # print(graph_idx) # 0, 1, 2, 3, 4
    return jsonify({"message": "버튼이 클릭되었습니다!"})


################################################### employee 전용 ###################################################
@bp.route('/employee_dash')
def employee_dash():
    return render_template("employee_dash.html")