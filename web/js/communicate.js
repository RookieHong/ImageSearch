function showSearchResult(matchList) {
    $('#searchResult').empty()

    for(var i = 0, length = matchList.length; i < length; i++) {
        $div = $('<div class="col-sm-4 col-md-4"></div>')

        $a = $('<a href="#" class="thumbnail"></a>')

        $img = $('<img style="width: 300px;height: 225px;"></img>')
        $img.attr('src', matchList[i][0])

        $label = $('<h5 class="text-center"></h5>')
        $label.text('Cosine distance: ' + Math.round(matchList[i][1] * 1000) / 1000)    //save 3 bits after dot

        $a.append($img)
        $div.append($a)
        $div.append($label)
        $('#searchResult').append($div)
    }
}

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
        res = JSON.parse(res)

        showSearchResult(res.matchList)

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