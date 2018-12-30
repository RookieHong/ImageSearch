$('.well').on('click', 'img', function() {
    imgPath = $(this).attr('src')
    $('#modalImage').attr('src', imgPath)
    $('#imageModal').modal()
})

$('#fileInput').on('change', function() {
    imgPath = $(this).val()
    ext = getFileExt(imgPath)
    if(ext == '') {
        $(this).val('')
        alert('Please select a image(.jpg, .jpeg, .gif, .jpeg)')
    }
})