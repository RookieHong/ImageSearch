$('.well').on('click', 'img', function() {  //Click an image to enlarge it
    imgPath = $(this).attr('src')
    $('#modalImage').attr('src', imgPath)
    $('#imageModal').modal()
})

$('#predictorsList').on('click', 'li', function() { //Change the text in the dropdown button when clicking an item in dropdown list
    predictor = $(this).children().text()
    $('#predictor').html(predictor + '<span class="caret">')
})

$('#fileInput').on('change', function() {   //Check if the uploaded file is an image
    imgPath = $(this).val()
    ext = getFileExt(imgPath)
    if(ext == '') {
        $(this).val('')
        alert('Please select a image(.jpg, .jpeg, .gif, .jpeg)')
    }
})