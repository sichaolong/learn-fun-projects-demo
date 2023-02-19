const { Configuration, OpenAIApi } = require('openai');
// open-ai配置
const configuration = new Configuration({
  organization: "你的openai的组织，登录open-ai后点击头像> View All Keys >即可看到 ",
  apiKey: "你的openai的key，登录open-ai后点击头像> View All Keys >即可看到",
})
const openai = new OpenAIApi(configuration);

// 文字搜索
const generateText = async (req, res) => {
  const { prompt } = req.body;



  try {


    const response = await openai.createCompletion({
      model: "text-davinci-003",
      prompt: prompt,
      temperature: 0.5,
      max_tokens: 1000,


    });


    // console.log(response.data.choices)

    const answer = response.data.choices[0].text;

    res.status(200).json({
      success: true,
      data: answer,
    });
  } catch (error) {
    if (error.response) {
      console.log(error.response.status);
      console.log(error.response.data);
    } else {
      console.log(error.message);
    }

    res.status(400).json({
      success: false,
      error: 'The answer could not be generated',
    });
  }
};

module.exports = { generateText };
