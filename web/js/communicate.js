function uploadImage() {
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
        $('#inputImg').attr('src', '../cgi/input.jpg')
        $('#outputImg').attr('src', '../cgi/output.jpg')
    }).fail(function(err) {
        console.log(err)
    });
}