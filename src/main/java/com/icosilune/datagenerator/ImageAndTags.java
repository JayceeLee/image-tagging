/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.icosilune.datagenerator;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableSet;
import java.awt.image.BufferedImage;

/**
 *
 * @author ashmore
 */
@AutoValue
public abstract class ImageAndTags {
  public abstract BufferedImage getImage();
  public abstract ImmutableSet<String> getTags();
  
  public static ImageAndTags create(BufferedImage image, Iterable<String> tags) {
    return new AutoValue_ImageAndTags(image, ImmutableSet.copyOf(tags));
  }
}
