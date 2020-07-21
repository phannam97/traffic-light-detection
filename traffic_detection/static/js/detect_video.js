$(function () {
    let socket = new WebSocket(
        'ws://'
        + window.location.host
        + '/ws/traffic_detection/'
        + 'video'
        + '/'
    );
    var close = false
    socket.onmessage = function (e) {
        if (e.data != '') {
            $(":loading").loading("stop");
        }
        document.querySelector('.initial-image').src = 'data:image/jpeg;base64,' + e.data;
    }
    socket.onclose = function (e) {
        console.error('Chat socket closed')
    }
    document.querySelector('#bnt-detect').onclick = function (e) {
        $("#loading-custom-animation").loading({
            onStart: function (loading) {
                loading.overlay.slideDown(400);
            },
            onStop: function (loading) {
                loading.overlay.slideUp(400);
            }
        });
        $('.initial-image').show();
        $('.box__video').hide();
        $("#drag-and-drop-zone").hide();
        socket.send(JSON.stringify({
            'message': 'video'
        }))
    }
    $('#id_videofile').attr('accept', 'video/*')
    $('#id_videofile').on("change", function () {
        if (this.files && this.files[0]) {
            $("#drag-and-drop-zone").hide();
            var $source = $('#video_here');
            $source[0].src = URL.createObjectURL(this.files[0]);
            $source.parent()[0].load();
            $('.box__video').show();
            $('#id_video').hide();

        }

    });
    $('.btn-refresh').on('click', function () {
        let video = document.querySelector('#video_here');
        video.removeAttribute('src')
        $('.initial-image').hide();
        $('#id_videofile').val('')
        $('.box__video').hide();
        $('#id_videofile').show();
        $("#drag-and-drop-zone").show()
        $.ajax({
            url: 'refresh',
            type: 'POST',
            cache: false,
            contentType: false,
            processData: false
        });
        close = true;
        socket.close()


    })
    $('#singlebutton').on('click', function (e) {
        e.preventDefault();
        var data = new FormData($('form').get(0));
        $.ajax({
            url: 'upload_video',
            type: 'POST',
            data: data,
            success: function (response) {
                console.log(response)
                if (response.error == false) {
                    $.notify("Upload Successful", "success");
                } else {
                    $.notify("Failed");
                }
                if(close == true){
                    socket = new WebSocket(
                        'ws://'
                        + window.location.host
                        + '/ws/traffic_detection/'
                        + 'video'
                        + '/'
                    );
                    close = false
                }
            },
            cache: false,
            contentType: false,
            processData: false
        });
    })

})
