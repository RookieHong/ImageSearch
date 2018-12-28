function uploadImage() {
    $('#upload').attr('disabled', 'disabled')
    $('#upload').text('processing')

    $('.alert').addClass('hide')
    $('.alert-warning').removeClass('hide')

    var formData = new FormData();
    formData.append('file', $('#fileInput')[0].files[0]);
    formData.append('ifSearch', $('#ifSearch').prop('checked'))
    formData.append('ifWholeImage', $('#ifWholeImage').prop('checked'))
    formData.append('ifBoundingBoxRegression', $('#ifBoundingBoxRegression').prop('checked'))
    $.ajax({
        url: '../cgi/process.py',
        type: 'POST',
        cache: false,
        data: formData,
        processData: false,
        contentType: false
    }).done(function(res) {
        console.log(res)

        $('.alert-warning').addClass('hide')
        $('.alert-success').removeClass('hide')

        $('#inputImg').attr('src', '../cgi/input.jpg?' + Math.random()) //makes src different every time, so the image shown will be changed when you upload more than once
        $('#outputImg').attr('src', '../cgi/output.jpg?' + Math.random())

        $('#upload').removeAttr('disabled')
        $('#upload').text('upload')
    }).fail(function(err) {
        console.log(err)

        $('.alert-warning').addClass('hide')
        $('.alert-danger').removeClass('hide')

        $('#upload').removeAttr('disabled')
        $('#upload').text('upload')
    });
}