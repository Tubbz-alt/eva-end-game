import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
from textwrap import wrap
from PIL import Image
from prediction import predict

def imshow(img, title=None, normalizeVal=0.5):
    img = img / 2 + normalizeVal     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)

def getDevice():
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  return device

def showImages(images, titles):
  fig3 = plt.figure(figsize = (25,15))
  for i, im in enumerate(images):
      sub = fig3.add_subplot(5, 5, i+1)
      plt.imshow(im[0].permute(1, 2, 0).cpu().numpy().squeeze(), interpolation='none')
      sub.set_title("\n".join(wrap(titles[i])))
  plt.tight_layout()
  plt.show()

def getPredActualTitle(output, classes):
  titles = []
  for im in output:
    titles.append("Prediction : %s, Actual: %s" % (classes[im[1].data.cpu().numpy()[0]], classes[im[2].data.cpu().numpy()[0]]))
  return titles

def getMisclassifiedImages(modelClass, test_loader, device, modelPath):
  model = modelClass
  model.load_state_dict(torch.load(modelPath))
  model.cuda()
  model.eval()
  misclassifiedImages = []
  with torch.no_grad():
      for data, target in test_loader:
          data, target = data.to(device), target.to(device)
          output = model(data)
          pred = output.argmax(dim=1, keepdim=True)
          target_modified = target.view_as(pred)
          for i in range(len(pred)):
            if pred[i].item()!= target_modified[i].item():
                misclassifiedImages.append([data[i], pred[i], target_modified[i]])
  return misclassifiedImages

def plotMisclassifiedImages(misclassifiedImages, classes, noOfImages=25):
  titles = getPredActualTitle(misclassifiedImages[:noOfImages], classes)
  showImages(misclassifiedImages[:noOfImages], titles)


def saveModel(model, modelPath):
  torch.save(model.state_dict(), modelPath)


def plot_endgame_images(data, batch_size):
  if batch_size > 8:
    batch_size = 8
  fig = plt.figure(figsize = (15,15))
  for i in range(batch_size):
      sub = fig.add_subplot(8, 4, (4*i)+1)
      plt.imshow(data["bg"][i].permute(1, 2, 0).cpu().numpy().squeeze(), interpolation='none')
      sub.set_title("BG")
      sub = fig.add_subplot(8, 4, (4*i)+2)
      plt.imshow(data["fg_bg"][i].permute(1, 2, 0).cpu().numpy().squeeze(), interpolation='none')
      sub.set_title("FG BG")
      sub = fig.add_subplot(8, 4, (4*i)+3)
      plt.imshow(data["fg_bg_mask"][i].cpu().numpy().squeeze(), interpolation='none')
      sub.set_title("FG BG Mask")
      sub = fig.add_subplot(8, 4, (4*i)+4)
      plt.imshow(data["dense_depth"][i].cpu().numpy().squeeze(), interpolation='none')
      sub.set_title("Dense Depth")
  plt.subplots_adjust(wspace=0, hspace=0)
  plt.tight_layout()
  plt.show()

def plot_endgame_predictions(data, prediction, batch_size=5):
  if batch_size > 8:
    batch_size = 8
  fig = plt.figure(figsize = (15,15))
  for i in range(batch_size):
    sub = fig.add_subplot(8, 6, (6*i)+1)
    plt.imshow(data["bg"][i].permute(1, 2, 0).cpu().numpy().squeeze(), interpolation='none')
    sub.set_title("BG")
    sub = fig.add_subplot(8, 6, (6*i)+2)
    plt.imshow(data["fg_bg"][i].permute(1, 2, 0).cpu().numpy().squeeze(), interpolation='none')
    sub.set_title("FG BG")
    sub = fig.add_subplot(8, 6, (6*i)+3)
    plt.imshow(data["fg_bg_mask"][i].cpu().numpy().squeeze(), interpolation='none')
    sub.set_title("FG BG Mask - Ground Truth")
    sub = fig.add_subplot(8, 6, (6*i)+4)
    plt.imshow(Image.fromarray(prediction["fg_bg_mask"][i]), interpolation='none')
    sub.set_title("FG BG Mask - Prediction")
    sub = fig.add_subplot(8, 6, (6*i)+5)
    plt.imshow(data["dense_depth"][i].cpu().numpy().squeeze(), interpolation='none')
    sub.set_title("Dense Depth - Ground Truth")
    sub = fig.add_subplot(8, 6, (6*i)+6)
    plt.imshow(Image.fromarray(prediction["dense_depth"][i]), interpolation='none')
    sub.set_title("Dense Depth - Prediction")
    
  plt.subplots_adjust(wspace=0, hspace=0)
  plt.tight_layout()
  plt.show()

def show_predictions(model, loader, device, noOfImages=8):
  data = next(iter(loader))
  dep_out, seg_out = predict(model, device, data)
  prediction = {"dense_depth":dep_out, "fg_bg_mask":seg_out}
  plot_endgame_predictions(data, prediction, 8)