function uploadImage() {
    $('#upload').attr('disabled', 'disabled')
    var formData = new FormData();
    formData.append('file', $('#file')[0].files[0]);
    console.log(formData)
    $.ajax({
        url: '../cgi/process.py',
        type: 'POST',
        cache: false,
        data: formData,
        processData: false,
        contentType: false
    }).done(function(res) {
        console.log(res)
        $('#inputImg').attr('src', '../cgi/input.jpg?' + Math.random()) //makes src different every time, so the image shown will be changed when you upload more than once
        $('#outputImg').attr('src', '../cgi/output.jpg?' + Math.random())
        $('#upload').removeAttr('disabled')
    }).fail(function(err) {
        console.log(err)
    });
}