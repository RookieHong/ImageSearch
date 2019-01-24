function getFileExt(filename)
{
    var flag = false;
    var ext = ''
    var arr = ["jpg", "png", "gif", "jpeg"];
    var index = filename.lastIndexOf(".");
    var ext = filename.substr(index+1).toLowerCase();
    for(var i = 0; i < arr.length; i++) {
        if(ext == arr[i]) {
            return arr[i]
        }
    }
    return ''
}

function enableButtons() {
    $('#upload').removeAttr('disabled')
    $('#upload').text('upload')

    $('#addToDB').removeAttr('disabled')
    $('#addToDB').text('addToDB')
}

function disableButtons() {
    $('#upload').attr('disabled', 'disabled')
    $('#upload').text('processing')

    $('#addToDB').attr('disabled', 'disabled')
    $('#addToDB').text('processing')
}

function showSearchResult(matchList) {
    $('#searchResult').empty()

    for(var i = 0, length = matchList.length; i < length; i++) {
        $div = $('<div class="col-sm-4 col-md-4"></div>')

        $a = $('<a href="javascript:;" class="thumbnail"></a>')

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

function uploadImage(ifAddImage) {
    disableButtons()

    $('.alert').addClass('hide')
    $('.alert-warning').removeClass('hide')

    ext = getFileExt($('#fileInput').val())

    var formData = new FormData();

    formData.append('file', $('#fileInput')[0].files[0]);
    formData.append('ext', ext)

    formData.append('searchType', $('#searchType').text())  //important
    formData.append('ifWholeImage', $('#ifWholeImage').prop('checked'))
    formData.append('ifAddImage', ifAddImage)
    formData.append('predictor', $('#predictor').text())
    $.ajax({
        url: '../cgi/process.py',
        type: 'POST',
        cache: false,
        data: formData,
        processData: false,
        contentType: false
    }).done(function(res) {
        res = JSON.parse(res)
        console.log(res.message)

        if(res.matchList) showSearchResult(res.matchList)

        status = res.status
        $('.alert-warning').addClass('hide')
        if(status == 'success') {
            $('.alert-success strong').text(res.message)
            $('.alert-success').removeClass('hide')
        }
        else {
            $('.alert-danger strong').text(res.message)
            $('.alert-danger').removeClass('hide')
        }

        if(ifAddImage == 'false') {
            $('#inputImg').attr('src', '../cgi/input.' + ext + '?' + Math.random()) //makes src different every time, so the image shown will be changed when you upload more than once
            $('#outputImg').attr('src', '../cgi/output.jpg?' + Math.random())
        }

        enableButtons()
    }).fail(function(err) {
        console.log(err)

        $('.alert-warning').addClass('hide')
        $('.alert-danger').removeClass('hide')

        enableButtons()
    });
}