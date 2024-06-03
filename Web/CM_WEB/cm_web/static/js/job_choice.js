console.log("직업 선택 페이지 -> 잘 연결됨")

// 관리자 선택 시 /dash로 이동 
document.querySelector("#supervisor_icon").onclick = function() {
    window.location.href = "/total";
};

// 작업자 선택 시 /employee_dash로 이동 
document.querySelector("#employee_icon").onclick = function() {
    window.location.href = "/employee_dash"; 
};