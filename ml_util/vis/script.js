"use strict;"

function ajax_get(ajax_conf) {
    var req = new XMLHttpRequest();
    req.open('GET', ajax_conf.url);
    req.onload = function() {

        if (req.status == 200) {
            ajax_conf.success(req.responseText);
        } else {
            ajax_conf.failure(req.status)
        }
    };

    req.send();
}

function toggle_log(rid, key) {
    var canvas_id = `log-canvas-${rid}-${key}`;
    var check_canvas = document.getElementById(canvas_id);
    if (check_canvas) {
        check_canvas.parentNode.removeChild(check_canvas);
        return;
    }

    var log_div = document.getElementById('log-div');
    ajax_get({
        url : `/get_log/${rid}`,
        success : function(log_json) {
            log_json = JSON.parse(log_json);

            var canvas = document.createElement('canvas');
            canvas.width = 800;
            canvas.height = 500;
            canvas.id = canvas_id;

            new Chart(canvas, log_json);

            log_div.appendChild(canvas);
        },
        failure : function(){}
    });
}


function create_run_div(run_info) {
    var frag = document.createDocumentFragment();

    var elem = document.createElement('div');
    elem.id = `select-run-${run_info.id}`;
    elem.class = 'select-run';
    elem.onclick = function(event) {
        toggle_log(run_info.id, 'accuracy');
    };
    frag.appendChild(elem);

    var head = document.createElement('h3');
    head.innerText = `${run_info.id}`;
    if (run_info.name !== undefined)
        head.innerText += `- ${run_info.name}`;
    elem.appendChild(head)
    
    if (run_info.desc !== undefined) {
        var desc = document.createElement('p');
        desc.innerText = run_info.desc;
        elem.appendChild(head)
    }

    return frag;

}

document.addEventListener("DOMContentLoaded", function () {
    var dir_div = document.getElementById('run-dir');

    ajax_get({
        url : '/get_runs',
        success : function(runs) {
            runs = JSON.parse(runs);
            for (var i = 0; i < runs.length; i++) {
                dir_div.appendChild(create_run_div(runs[i]));
            }
        },
        failure : function(){}
    });


});

//chart example
/*
document.addEventListener("DOMContentLoaded", function () {
    canvas = document.getElementById("log-chart").getContext("2d");

    ajax_get({
        url : '/get_log',
        success : function(rtext) {
            response = JSON.parse(rtext);
            new Chart(canvas, response);
        },
        failure : function(){}
    });

});

*/

