const { Configuration, OpenAIApi } = require('openai');

// open-ai配置
const configuration = new Configuration({
  organization: "你的openai的组织，登录open-ai后点击头像> View All Keys >即可看到 ",
  apiKey: "你的openai的key，登录open-ai后点击头像> View All Keys >即可看到",
})
const openai = new OpenAIApi(configuration);

// 生成图片
const generateImage = async (req, res) => {
  const { prompt, size } = req.body;

  const imageSize =
    size === 'small' ? '256x256' : size === 'medium' ? '512x512' : '1024x1024';

  try {
    const response = await openai.createImage({
      prompt,
      n: 1,
      size: imageSize,
    });
    console.log(response.data)

    const imageUrl = response.data.data[0].url;

    res.status(200).json({
      success: true,
      data: imageUrl,
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
      error: 'The image could not be generated',
    });
  }
};

module.exports = { generateImage };
