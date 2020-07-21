$(function () {
    $('.btn-refresh').on('click', function () {
        var img = document.querySelector('.initial-image');
        img.removeAttribute('src')
        $('.initial-image').hide();
        $('.odd').remove();
        $("#drag-and-drop-zone").show()
        $.ajax({
            url: '',
            type: 'POST',
            cache: false,
            contentType: false,
            processData: false
        });
    })
    $('#id_image').on('change', function (e) {
        e.preventDefault();
        if (this.files && this.files[0]) {
            $("#drag-and-drop-zone").hide();
            var img = document.querySelector('.initial-image');
            img.src = URL.createObjectURL(this.files[0]);
            $('.initial-image').show();
        }
    });
    $('#singlebutton').on('click', function (e) {
        e.preventDefault();
        var data = new FormData($('form').get(0));
        $.ajax({
            url: 'upload_image',
            type: 'POST',
            data: data,
            success: function (response) {
                console.log(response)
                if (response.error == false) {
                    $.notify("Upload Successful", "success");
                } else {
                    $.notify("Failed");
                }
            },
            cache: false,
            contentType: false,
            processData: false
        });
    })
    $('.btn-yolo').on('click', function () {
        $("#loading-custom-animation").loading({
            onStart: function (loading) {
                loading.overlay.slideDown(400);
            },
            onStop: function (loading) {
                loading.overlay.slideUp(400);
            }
        });
        $.ajax({
            url: 'detect-yolov4',
            type: 'POST',
            success: function (response) {
                console.log(response)
                var img = document.querySelector('.initial-image');
                img.src = response.url;
                $(":loading").loading("stop");
                for (var i = 0; i < response.boxes.length - 1; i++) {
                    $('.objects').append('<tr role="row" class="odd">' +
                        '<td class="" style="background:' + response.boxes[response.boxes.length - 1][i] + '">' + response.boxes[i].class + '</td>' +
                        '<td class="">' + response.boxes[i].confidence + '</td>' +
                        +'</tr>')
                }
            },
            cache: false,
            contentType: false,
            processData: false
        });
    })
    $('.btn-img').on('click', function () {
        $("#loading-custom-animation").loading({
            onStart: function (loading) {
                loading.overlay.slideDown(400);
            },
            onStop: function (loading) {
                loading.overlay.slideUp(400);
            }
        });
        $.ajax({
            url: 'detect-traffic',
            type: 'POST',
            success: function (response) {
                console.log(response)
                var img = document.querySelector('.initial-image');
                img.src = response.url;
                $(":loading").loading("stop");
                for (var i = 0; i < response.boxes.length - 1; i++) {
                    $('.objects').append('<tr role="row" class="odd">' +
                        '<td class=""">' + response.color_set[i] + '</td>' +
                        '<td class="">' + response.boxes[i].confidence + '</td>' +
                        +'</tr>')
                }
            },
            cache: false,
            contentType: false,
            processData: false
        });
    })
})

// function getRandomColor() {
//     var letters = '0123456789ABCDEFabcdefgthy';
//     var color = '#';
//     for (var i = 0; i < 6; i++) {
//         color += letters[Math.floor(Math.random() * 16)];
//     }
//     return color;
// }

