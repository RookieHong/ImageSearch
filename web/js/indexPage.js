$('.well').on('click', 'img', function() {  //Click an image to enlarge it
    imgPath = $(this).attr('src')
    $('#modalImage').attr('src', imgPath)
    $('#imageModal').modal()
})

$('ul.dropdown-menu').on('click', 'li', function() { //Change the text in the dropdown button when clicking an item in dropdown list
    text = $(this).children().text()
    textElement = $(this).parent().prev()
    $(textElement).html(text + '<span class="caret">')

    if(text == 'objects') {
        $('.objects').show()
        $('.wholeImage').hide()

        $('#predictor').html('resnet101<span class="caret">')
    }
    else if(text == 'wholeImage') {
        $('.objects').hide()
        $('.wholeImage').show()
    }
})

$('#fileInput').on('change', function() {   //Check if the uploaded file is an image
    imgPath = $(this).val()
    ext = getFileExt(imgPath)
    if(ext == '') {
        $(this).val('')
        alert('Please select a image(.jpg, .jpeg, .gif, .jpeg)')
    }
})

$('.objects').hide()
$('.wholeImage').show()