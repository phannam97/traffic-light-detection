$(function () {
    const socket = new WebSocket(
        'ws://'
        + window.location.host
        + '/ws/traffic_detection/'
        + 'camera'
        + '/'
    )
    socket.onmessage = function (e) {
        if (e.data != '') {
            $("#drag-and-drop-zone").hide();
            $('.initial-image').show();
            $(":loading").loading("stop");
        }
        document.querySelector('.initial-image').src = 'data:image/jpeg;base64,'+ e.data;

    }
    socket.onclose = function (e) {
        console.error('Chat socket closed')
    }
    $('#r1').on('click', function () {
        document.getElementById('ip-webcam').disabled = false;
    });

    $('#r2').on('click', function () {
        document.getElementById('ip-webcam').disabled = true;
    });
    $('#btn-connect').on('click', function (e) {
        $("#loading-custom-animation").loading({
            onStart: function (loading) {
                loading.overlay.slideDown(400);
            },
            onStop: function (loading) {
                loading.overlay.slideUp(400);
            }
        });


        if (document.getElementById('r1').checked) {
            option_value = document.getElementById('r1').value;
        } else {
            option_value = document.getElementById('r2').value;
        }
        var data;
        e.preventDefault();
        if (option_value == 'camera') {
            data = 0;
        } else {
            data = $('#ip-webcam').val();
        }

        socket.send(JSON.stringify({
            'message': data
        }))
    })

})
