
// express 사용
var express = require('express');
var app = express();
// body-parser 사용
var bodyParser = require('body-parser');
// post 방식 body 파싱하고 위해 사용
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// 서버 확인 코드 
app.get('/', function (req, res) {
  res.send('Hello World!');
});

// http 통신용 특정 포트 할당 및 포트 모니터링
app.listen(10001, function () {
  console.log('Example app listening on port 10001!');
});


// 카카오 API 서버로 부터 API 테스트 할때 들어 오는 부분 
app.get('/keyboard', (req, res) => {
  console.log('카카오 테스트 메시지');
   const message = {
		type: 'text',
		};
  res.set({
      'content-type': 'application/json'
  }).send(JSON.stringify(message));
});

// 카카오 메시지들이 들어 오는 부분
app.post('/message', (req, res) => {
     const _obj = {
        user_key: req.body.user_key,
        type: req.body.type,
        content: req.body.content
    };

    console.log(_obj.content)
	
	// 아래는 룰베이스로 통신 확인용 챗봇
    var message;
    if(_obj.content == '안녕') {
       message = {
            "message": {
		"text": '안녕하세요 DeepNLP 챗봇입니다'
	    },
            "keyboard": {
                "type": "buttons",
                "buttons": [
                    "배고프다",
                    "심심해"
              ]
            }
       };

       res.set({
          'content-type': 'application/json'
       }).send(JSON.stringify(message));
   } else if(_obj.content == '배고프다') {
	message = {
		"message": {
			"text": '편의점 도시락 먹오'
		},
		"keyboard": {
			"type": "buttons",
			"buttons": [
				"아니 짜장면 먹으러 갈래",
				"응 고마워"
			]
		}
	};

	res.set({
		'content-type': 'application/json'
	}).send(JSON.stringify(message));
   } else if(_obj.content == '심심해') {
	message = {
		"message": {
			"text": '끝말잇기 할까요?'
		},
		"keyboard": {
			"type": "buttons",
			"buttons": [
				"아니 그냥 잘래",
				"ㅋㅋㅋㅋㅋㅋ"
			]
		}
	};

	res.set({
		'content-type': 'application/json'
	}).send(JSON.stringify(message));
   } else {
	message = {
		"message": {
			"text": '사랑합니다. 다음에 뵈요.'
		}
	};

	res.set({
		'content-type': 'application/json'
	}).send(JSON.stringify(message));
   }		
});


