document.addEventListener("DOMContentLoaded", function() {
    var graphJSON = JSON.parse(document.getElementById('accumulate-data').textContent);
    Plotly.newPlot('accumulate_graph', graphJSON.data, graphJSON.layout);
});

