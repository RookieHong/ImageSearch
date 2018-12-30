$('.well').on('click', 'img', function() {
    imgPath = $(this).attr('src')
    $('#modalImage').attr('src', imgPath)
    $('#imageModal').modal()
})