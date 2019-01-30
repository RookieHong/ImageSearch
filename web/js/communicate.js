function getFileExt(filename)
{
    var flag = false;
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
    $('.upload').removeAttr('disabled')
    $('.upload').text('upload')

    $('.addToDB').removeAttr('disabled')
    $('.addToDB').text('addToDB')
}

function disableButtons() {
    $('.upload').attr('disabled', 'disabled')
    $('.upload').text('processing')

    $('.addToDB').attr('disabled', 'disabled')
    $('.addToDB').text('processing')
}

ext = ''

matchList_wholeImage = []
matchList_allObjects = []
matchList_singleObject = []

showCount_wholeImage = 0
showCount_objects = 0

function showSearchResult(tabPage, empty) {
    if(empty) $('#searchResult_' + tabPage).empty()

    if(tabPage == 'wholeImage') {
        matchList = matchList_wholeImage
        showCount = showCount_wholeImage
    }
    else {
        matchList = matchList_singleObject
        showCount = showCount_objects
    }

    for(j = 0, length = matchList.length; showCount < length && j < 9; showCount++, j++) {
        $div = $('<div class="col-sm-4 col-md-4"></div>')

        $a = $('<a href="javascript:;" class="thumbnail"></a>')

        $img = $('<img style="width: 300px;height: 225px;"></img>')
        $img.attr('src', matchList[showCount][0])

        $label = $('<h5 class="text-center"></h5>')
        $label.text('Cosine distance: ' + Math.round(matchList[showCount][1] * 1000) / 1000)    //save 3 bits after dot

        $a.append($img)
        $div.append($a)
        $div.append($label)
        $('#searchResult_' + tabPage).append($div)
    }

    if(tabPage == 'wholeImage') showCount_wholeImage = showCount
    else showCount_objects = showCount

    if(showCount >= matchList.length) $('#showMore_' + tabPage).addClass('hide')
}

function showObjects(objects) {
    var img = document.getElementById('inputImg_objects')
    $(img).load(function() {
        $('#segmentedObjects').empty()
        for(var i = 0; i < objects.length; i++) {
            var canvas = document.createElement('canvas')
            var ctx = canvas.getContext('2d')

            canvas.width = 300
            canvas.height = 150

            var x1 = objects[i].x1
            var y1 = objects[i].y1
            var width = objects[i].x2 - objects[i].x1
            var height = objects[i].y2 - objects[i].y1

            //x1 = x1 / img.naturalWidth * canvas.width
            //y1 = y1 / img.naturalHeight * canvas.height
            //width = width / img.naturalWidth * canvas.width
            //height = height / img.naturalHeight * canvas.height

            ctx.drawImage(img, x1, y1, width, height, 0, 0, canvas.width, canvas.height)

            $div = $('<div class="col-sm-4 col-md-4"></div>')
            $a = $('<a href="javascript:;" class="thumbnail object" num="' + objects[i].num + '"></a>')
            $label = $('<h5 class="text-center"></h5>')
            $label.text(objects[i].label)
            $a.append(canvas)
            $div.append($a)
            $div.append($label)
            $('#segmentedObjects').append($div)
        }
    })
}

function uploadImage_wholeImage(ifAddImage) {
    disableButtons()

    $('.alert').addClass('hide')
    $('.alert-warning').removeClass('hide')

    ext = getFileExt($('#fileInput_wholeImage').val())

    var formData = new FormData();

    formData.append('file', $('#fileInput_wholeImage')[0].files[0]);
    formData.append('ext', ext)

    formData.append('searchType', 'wholeImage')  //important
    formData.append('ifAddImage', ifAddImage)
    formData.append('predictor', $('#predictor_wholeImage').text())
    $.ajax({
        url: '../cgi/process.py',
        type: 'POST',
        cache: false,
        data: formData,
        processData: false,
        contentType: false
    }).done(function(res) {
        res = JSON.parse(res)
        console.log(res)

        if(res.matchList) {
            showCount_wholeImage = 0
            matchList_wholeImage = res.matchList
            $('#showMore_wholeImage').removeClass('hide')
            showSearchResult('wholeImage', true)
        }

        status = res.status
        $('.alert-warning').addClass('hide')
        if(status == 'success') {
            $('#successMsg_wholeImage strong').text(res.message)
            $('.alert-success').removeClass('hide')
        }
        else {
            $('#dangerMsg_wholeImage strong').text(res.message)
            $('.alert-danger').removeClass('hide')
        }

        if(ifAddImage == 'false') {
            $('#inputImg_wholeImage').attr('src', '../cgi/input.' + ext + '?' + Math.random()) //makes src different every time, so the image shown will be changed when you upload more than once
            $('#classification_wholeImage').text(res.classification)
        }

        enableButtons()
    }).fail(function(err) {
        console.log(err)

        $('.alert-warning').addClass('hide')
        $('.alert-danger').removeClass('hide')

        enableButtons()
    });
}

function uploadImage_objects(ifAddImage) {
    disableButtons()

    $('.alert').addClass('hide')
    $('.alert-warning').removeClass('hide')

    ext = getFileExt($('#fileInput_objects').val())

    var formData = new FormData();

    formData.append('file', $('#fileInput_objects')[0].files[0]);
    formData.append('ext', ext)

    formData.append('searchType', 'objects')  //important
    formData.append('ifAddImage', ifAddImage)
    formData.append('predictor', $('#predictor_objects').text())
    $.ajax({
        url: '../cgi/process.py',
        type: 'POST',
        cache: false,
        data: formData,
        processData: false,
        contentType: false
    }).done(function(res) {
        res = JSON.parse(res)
        console.log(res)

        if(res.matchList) {
            matchList_objects = res.matchList
        }

        if(res.objects) {
            showObjects(res.objects)
            $('.well').on('click', 'a.object', function () {    //Switch global value matchList when clicking on an object
              num = parseInt($(this).attr('num'))
              matchList_singleObject = matchList_objects[num]
              showCount_objects = 0
              $('#showMore_objects').removeClass('hide')
              showSearchResult('objects', true)
            })
        }

        status = res.status
        $('.alert-warning').addClass('hide')
        if(status == 'success') {
            $('#successMsg_objects strong').text(res.message)
            $('.alert-success').removeClass('hide')
        }
        else {
            $('#dangerMsg_objects strong').text(res.message)
            $('.alert-danger').removeClass('hide')
        }

        if(ifAddImage == 'false') {
            $('#inputImg_objects').attr('src', '../cgi/input.' + ext + '?' + Math.random()) //makes src different every time, so the image shown will be changed when you upload more than once
            $('#outputImg_objects').attr('src', '../cgi/output.jpg?' + Math.random())
        }

        enableButtons()
    }).fail(function(err) {
        console.log(err)

        $('.alert-warning').addClass('hide')
        $('.alert-danger').removeClass('hide')

        enableButtons()
    });
}